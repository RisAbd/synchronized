"""Вьюхи сервиса: библиотека, добавление по ссылке, плеер, data.json, аудио (Range), статус.

Каждая запись может иметь несколько прогонов ASR (по распознавателям) — для сравнения точности.
Активный прогон плеера выбирается по ?asr=<recognizer> либо авто (приоритет в recognizers.PRIORITY).
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from django.conf import settings
from django.http import (FileResponse, Http404, HttpResponse, HttpResponseRedirect,
                         JsonResponse, StreamingHttpResponse)
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from . import pipeline, recognizers
from .models import AsrRun, Recitation
from .tasks import dispatch, dispatch_run


def index(request):
    recs = Recitation.objects.prefetch_related("runs").all()
    q = (request.GET.get("q") or "").strip()
    if q:
        # поиск по названию/чтецу/ссылке/названиям сур; surahs_label — вычисляемое, фильтруем в Python
        recs = [r for r in recs if r.matches_query(q)]
    return render(request, "recitations/index.html", {
        "recs": recs,
        "q": q,
        "recognizers": recognizers.selectable_recognizers(),
        "default_recognizers": settings.DEFAULT_RECOGNIZERS,
    })


def _chosen_recognizers(request) -> list[str]:
    """Список распознавателей из формы (чекбоксы), с фолбэком на дефолт.
    Выравниватели (forced) сюда не берём — они запускаются автоматически пост-шагом."""
    def ok(r):
        return recognizers.is_valid(r) and not recognizers.is_aligner(r)
    chosen = [r for r in request.POST.getlist("recognizers") if ok(r)]
    if not chosen:
        chosen = [r for r in settings.DEFAULT_RECOGNIZERS if ok(r)]
    return chosen or ["whisper"]


@require_POST
def add(request):
    url = (request.POST.get("source_url") or "").strip()
    if not url:
        return HttpResponseRedirect(reverse("index"))
    # П7: уже парсили эту ссылку → не создаём дубль, ведём на существующую запись
    dup = Recitation.find_by_source(url)
    if dup:
        return HttpResponseRedirect(reverse("player", args=[dup.id]))
    is_yt = bool(re.search(r"(youtube\.com|youtu\.be)", url))
    rec = Recitation.objects.create(
        source_url=url,
        source_type="youtube" if is_yt else ("file" if os.path.exists(url) else "other"),
        title=(request.POST.get("title") or "").strip(),
        reciter=(request.POST.get("reciter") or "").strip(),
        gstt_key=(request.POST.get("gstt_key") or "").strip(),
        status=Recitation.Status.QUEUED,
    )
    for rkey in _chosen_recognizers(request):
        AsrRun.objects.create(recitation=rec, recognizer=rkey, status=AsrRun.Status.QUEUED)
    dispatch(rec.id)
    return HttpResponseRedirect(reverse("index"))


@require_POST
def run(request, pk):
    """Добавить/пересчитать один распознаватель для существующей записи."""
    rec = get_object_or_404(Recitation, pk=pk)
    rkey = (request.POST.get("recognizer") or "").strip()
    # forced align — не выбираемый распознаватель, а авто-пост-шаг после ASR (не запускаем вручную)
    if not recognizers.is_valid(rkey) or recognizers.is_aligner(rkey):
        return HttpResponseRedirect(reverse("player", args=[pk]))
    run_obj, _ = AsrRun.objects.get_or_create(recitation=rec, recognizer=rkey)
    run_obj.status = AsrRun.Status.QUEUED
    run_obj.error = ""
    run_obj.save(update_fields=["status", "error", "updated_at"])
    dispatch_run(run_obj.id)
    return HttpResponseRedirect(reverse("player", args=[pk]) + f"?asr={rkey}")


@require_POST
def delete(request, pk):
    rec = get_object_or_404(Recitation, pk=pk)
    # чистим папку записи целиком (аудио + сырые ASR-выгрузки)
    d = pipeline.rec_dir(rec.id)
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
    # legacy-аудио демо в web/audio (кроме локальных исходников — их не трогаем)
    if rec.audio_filename and rec.source_type != "file":
        legacy = Path(settings.AUDIO_DIR) / rec.audio_filename
        if legacy.is_file():
            try:
                legacy.unlink()
            except OSError:
                pass
    rec.delete()
    return HttpResponseRedirect(reverse("index"))


def player(request, pk):
    """Плеер теперь СТАТИЧНЫЙ файл (П11): бэк HTML не отдаёт. Красивый URL /r/<id>/ ведём
    редиректом на статику, пробрасывая ?asr/?debug (для старых ссылок/закладок). Статика
    сама фетчит /r/<id>/data.json и рисует."""
    get_object_or_404(Recitation, pk=pk)
    qs = request.GET.urlencode()
    url = f"{settings.STATIC_URL}player.html?rec={pk}" + (f"&{qs}" if qs else "")
    return redirect(url)


def api_recitations(request):
    """JSON-список записей для статичной библиотеки (П11): фронт сам фетчит и рисует."""
    items = []
    for rec in Recitation.objects.prefetch_related("runs").order_by("-id"):
        items.append({
            "id": rec.id, "title": rec.title or f"Запись #{rec.id}",
            "title_ar": rec.title_ar, "reciter": rec.reciter,
            "source_url": rec.source_url, "status": rec.status,
            "ready": rec.is_ready, "youtube_id": rec.youtube_id,
            "duration": rec.duration, "meta": rec.meta or {},
            "runs": [{"recognizer": r.recognizer, "label": r.label,
                      "is_aligner": r.is_aligner, "status": r.status,
                      "metrics": r.metrics or {}} for r in rec.runs.all()],
        })
    resp = JsonResponse({"recitations": items})
    resp["Access-Control-Allow-Origin"] = "*"
    return resp


def data_json(request, pk):
    rec = get_object_or_404(Recitation, pk=pk)
    prefer = request.GET.get("asr")
    run = rec.active_run(prefer)
    data = run.data if (run and run.data) else rec.data
    if not data:
        raise Http404("нет данных")
    payload = dict(data)
    # прогоны — для отладочного тумблера распознавания/выравнивания в СТАТИЧНОМ плеере
    # (раньше рисовались server-side в шаблоне; теперь фронт строит их из JSON сам).
    runs_payload = [
        {"recognizer": r.recognizer, "label": r.label, "is_aligner": r.is_aligner,
         "status": r.status, "stage": r.stage, "error": (r.error or "")[:300],
         "metrics": r.metrics or {}}
        for r in rec.runs.all()
    ]
    payload.update({"id": rec.id, "title": rec.title or f"Запись #{rec.id}",
                    "title_ar": rec.title_ar, "reciter": rec.reciter,
                    "recognizer": run.recognizer if run else None,
                    "metrics": (run.metrics if run else None) or {},
                    "youtube_id": rec.youtube_id,
                    "runs": runs_payload,
                    "active_key": run.recognizer if run else None,
                    "audio": reverse("audio", args=[rec.id])})
    resp = JsonResponse(payload)
    # разрешаем фетч со статичного фронта на другом origin (будущий GitHub Pages, П11)
    resp["Access-Control-Allow-Origin"] = "*"
    return resp


def card(request, pk):
    """HTML одной карточки списка — для точечной перерисовки на фронте (index.html опрашивает
    /status и при смене статуса заменяет ТОЛЬКО эту карточку, без перезагрузки страницы)."""
    rec = get_object_or_404(Recitation.objects.prefetch_related("runs"), pk=pk)
    return render(request, "recitations/_card.html", {"r": rec})


def status(request, pk):
    rec = get_object_or_404(Recitation, pk=pk)
    # строка шага: если какой-то прогон обрабатывается — покажем «<распознаватель>: <шаг>»
    stage = rec.stage
    proc = [r for r in rec.runs.all() if r.status == Recitation.Status.PROCESSING]
    if proc:
        stage = f"{proc[0].label}: {proc[0].stage or '…'}"
    return JsonResponse({
        "status": rec.status, "stage": stage,
        "ready": rec.is_ready, "error": rec.error[:400],
        "runs": [{"recognizer": r.recognizer, "label": r.label, "status": r.status,
                  "stage": r.stage, "error": (r.error or "")[:300],
                  "metrics": r.metrics or {}} for r in rec.runs.all()],
    })


def audio(request, pk):
    """Отдача аудио с поддержкой HTTP Range (перемотка/стриминг)."""
    rec = get_object_or_404(Recitation, pk=pk)
    if not rec.audio_filename:
        raise Http404("нет аудио")
    path = pipeline.rec_dir(rec.id) / rec.audio_filename
    if not path.is_file():
        path = Path(settings.AUDIO_DIR) / rec.audio_filename  # legacy демо
    if not path.is_file():
        raise Http404("файл не найден")

    size = path.stat().st_size
    ctype = "audio/mpeg" if path.suffix in (".mp3", ".mpeg") else "application/octet-stream"
    rng = request.headers.get("Range")
    if not rng:
        resp = FileResponse(open(path, "rb"), content_type=ctype)
        resp["Content-Length"] = str(size)
        resp["Accept-Ranges"] = "bytes"
        return resp

    m = re.match(r"bytes=(\d*)-(\d*)", rng)
    start = int(m.group(1)) if m and m.group(1) else 0
    end = int(m.group(2)) if m and m.group(2) else size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        r = HttpResponse(status=416)
        r["Content-Range"] = f"bytes */{size}"
        return r

    length = end - start + 1

    def chunks():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                buf = f.read(min(64 * 1024, remaining))
                if not buf:
                    break
                remaining -= len(buf)
                yield buf

    resp = StreamingHttpResponse(chunks(), status=206, content_type=ctype)
    resp["Content-Range"] = f"bytes {start}-{end}/{size}"
    resp["Content-Length"] = str(length)
    resp["Accept-Ranges"] = "bytes"
    return resp
