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
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import pipeline, recognizers
from .models import AsrRun, Recitation
from .tasks import dispatch, dispatch_run


def index(request):
    """Библиотека теперь тоже СТАТИЧНАЯ (П11): бэк HTML не отдаёт. Корень / ведём редиректом на
    статику, которая фетчит /api/recitations и рисует список сама (поиск — на клиенте)."""
    qs = request.GET.urlencode()
    return redirect(f"{settings.STATIC_URL}index.html" + (f"?{qs}" if qs else ""))


def _chosen_recognizers(request) -> list[str]:
    """Список распознавателей из формы (чекбоксы), с фолбэком на дефолт.
    Выравниватели (forced) сюда не берём — они запускаются автоматически пост-шагом."""
    def ok(r):
        return recognizers.is_valid(r) and not recognizers.is_aligner(r)
    chosen = [r for r in request.POST.getlist("recognizers") if ok(r)]
    if not chosen:
        chosen = [r for r in settings.DEFAULT_RECOGNIZERS if ok(r)]
    return chosen or ["whisper"]


# CSRF-exempt: add/delete зовутся fetch-ом из СТАТИЧНОГО фронта (П11), в т.ч. потенциально с
# другого origin (GitHub Pages) — CSRF-cookie туда не доедет. Это персональный прототип на одного
# пользователя; при мультиюзере вернуть токен/аутентификацию.
@csrf_exempt
@require_POST
def add(request):
    url = (request.POST.get("source_url") or "").strip()
    if not url:
        return _cors(JsonResponse({"ok": False, "error": "пустая ссылка"}, status=400))
    # П7: уже парсили эту ссылку → не создаём дубль, ведём на существующую запись
    dup = Recitation.find_by_source(url)
    if dup:
        return _cors(JsonResponse({"ok": True, "id": dup.id, "dup": True}))
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
    return _cors(JsonResponse({"ok": True, "id": rec.id}))


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


@csrf_exempt
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
    return _cors(JsonResponse({"ok": True}))


def player(request, pk):
    """Плеер теперь СТАТИЧНЫЙ файл (П11): бэк HTML не отдаёт. Красивый URL /r/<id>/ ведём
    редиректом на статику, пробрасывая ?asr/?debug (для старых ссылок/закладок). Статика
    сама фетчит /r/<id>/data.json и рисует."""
    get_object_or_404(Recitation, pk=pk)
    qs = request.GET.urlencode()
    url = f"{settings.STATIC_URL}player.html?rec={pk}" + (f"&{qs}" if qs else "")
    return redirect(url)


def manual(request, pk):
    """Ручной элайнер (П12) — тоже статичный файл. Красивый URL /r/<id>/manual ведём
    редиректом на статику (пробрасываем ?asr/?api). Страница фетчит /r/<id>/data.json."""
    get_object_or_404(Recitation, pk=pk)
    qs = request.GET.urlencode()
    url = f"{settings.STATIC_URL}manual.html?rec={pk}" + (f"&{qs}" if qs else "")
    return redirect(url)


def _run_dict(r):
    m = r.metrics or {}
    cov = m.get("coverage")
    return {"recognizer": r.recognizer, "label": r.label, "is_aligner": r.is_aligner,
            "status": r.status, "status_display": r.get_status_display(),
            "stage": r.stage, "error": (r.error or "")[:300],
            "wt": m.get("wt") or 0, "coverage_pct": round(cov * 100) if cov else None,
            "speech_sec": m.get("speech_sec"), "silence_sec": m.get("silence_sec"),
            "duration_sec": m.get("duration")}


def _rec_dict(rec):
    """Готовые к показу поля записи для статичной библиотеки (вычисляемые свойства — на беке,
    чтобы фронт не тянул quran.db и не считал длительности/названия сур)."""
    return {
        "id": rec.id, "title": rec.title or f"Запись #{rec.id}",
        "title_ar": rec.title_ar, "reciter": rec.reciter,
        "source_url": rec.source_url, "status": rec.status,
        "status_display": rec.get_status_display(),
        "ready": rec.is_ready, "stage": rec.stage, "error": (rec.error or "")[:300],
        "youtube_id": rec.youtube_id, "duration": rec.duration,
        "thumbnail": rec.thumbnail, "surahs_label": rec.surahs_label,
        "duration_h": rec.duration_h, "filesize_h": rec.filesize_h, "ext_h": rec.ext_h,
        "has_active_runs": rec.has_active_runs,
        "runs": [_run_dict(r) for r in rec.runs.all()],
    }


def _cors(resp):
    resp["Access-Control-Allow-Origin"] = "*"
    return resp


def api_recitations(request):
    """JSON-список записей для статичной библиотеки (П11): фронт сам фетчит и рисует."""
    items = [_rec_dict(rec) for rec in
             Recitation.objects.prefetch_related("runs").order_by("-id")]
    return _cors(JsonResponse({
        "recitations": items,
        "recognizers": [{"key": r.key, "label": r.label, "note": r.note}
                        for r in recognizers.selectable_recognizers()],
        "default_recognizers": list(settings.DEFAULT_RECOGNIZERS),
    }))


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
