"""Вьюхи сервиса: библиотека, добавление по ссылке, плеер, data.json, аудио (Range), статус."""
from __future__ import annotations

import os
import re
from pathlib import Path

from django.conf import settings
from django.http import (FileResponse, Http404, HttpResponse, HttpResponseRedirect,
                         JsonResponse, StreamingHttpResponse)
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .models import Recitation
from .tasks import dispatch


def index(request):
    recs = Recitation.objects.all()
    return render(request, "recitations/index.html", {"recs": recs})


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
        status=Recitation.Status.QUEUED,
    )
    dispatch(rec.id)
    return HttpResponseRedirect(reverse("index"))


@require_POST
def delete(request, pk):
    rec = get_object_or_404(Recitation, pk=pk)
    if rec.audio_filename:
        f = Path(settings.AUDIO_DIR) / rec.audio_filename
        if f.is_file():
            try:
                f.unlink()
            except OSError:
                pass
    rec.delete()
    return HttpResponseRedirect(reverse("index"))


def player(request, pk):
    rec = get_object_or_404(Recitation, pk=pk)
    return render(request, "recitations/player.html", {"rec": rec})


def data_json(request, pk):
    rec = get_object_or_404(Recitation, pk=pk)
    if not rec.data:
        raise Http404("нет данных")
    payload = dict(rec.data)
    payload.update({"id": rec.id, "title": rec.title or f"Запись #{rec.id}",
                    "title_ar": rec.title_ar, "reciter": rec.reciter,
                    "audio": reverse("audio", args=[rec.id])})
    return JsonResponse(payload)


def status(request, pk):
    rec = get_object_or_404(Recitation, pk=pk)
    return JsonResponse({"status": rec.status, "stage": rec.stage,
                         "ready": rec.is_ready, "error": rec.error[:400]})


def audio(request, pk):
    """Отдача аудио с поддержкой HTTP Range (перемотка/стриминг)."""
    rec = get_object_or_404(Recitation, pk=pk)
    if not rec.audio_filename:
        raise Http404("нет аудио")
    path = Path(settings.AUDIO_DIR) / rec.audio_filename
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
