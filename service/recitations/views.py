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
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from . import pipeline, recognizers
from .models import AsrRun, Recitation
from .tasks import dispatch, dispatch_run


def index(request):
    recs = Recitation.objects.prefetch_related("runs").all()
    return render(request, "recitations/index.html", {
        "recs": recs,
        "recognizers": recognizers.all_recognizers(),
        "default_recognizers": settings.DEFAULT_RECOGNIZERS,
    })


def _chosen_recognizers(request) -> list[str]:
    """Список распознавателей из формы (чекбоксы), с фолбэком на дефолт."""
    chosen = [r for r in request.POST.getlist("recognizers") if recognizers.is_valid(r)]
    if not chosen:
        chosen = [r for r in settings.DEFAULT_RECOGNIZERS if recognizers.is_valid(r)]
    return chosen or ["whisper"]


@require_POST
def add(request):
    url = (request.POST.get("source_url") or "").strip()
    if not url:
        return HttpResponseRedirect(reverse("index"))
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
    if not recognizers.is_valid(rkey):
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
    rec = get_object_or_404(Recitation, pk=pk)
    prefer = request.GET.get("asr")
    active = rec.active_run(prefer)
    active_key = active.recognizer if active else None
    return render(request, "recitations/player.html", {
        "rec": rec,
        "active_key": active_key,
        "runs": rec.runs.all(),
        "youtube_id": rec.youtube_id,
    })


def data_json(request, pk):
    rec = get_object_or_404(Recitation, pk=pk)
    prefer = request.GET.get("asr")
    run = rec.active_run(prefer)
    data = run.data if (run and run.data) else rec.data
    if not data:
        raise Http404("нет данных")
    payload = dict(data)
    payload.update({"id": rec.id, "title": rec.title or f"Запись #{rec.id}",
                    "title_ar": rec.title_ar, "reciter": rec.reciter,
                    "recognizer": run.recognizer if run else None,
                    "metrics": (run.metrics if run else None) or {},
                    "youtube_id": rec.youtube_id,
                    "audio": reverse("audio", args=[rec.id])})
    return JsonResponse(payload)


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
