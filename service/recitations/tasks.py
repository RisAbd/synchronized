"""Фоновая обработка записи: Celery-задачи + диспетчер.

В docker-compose работает через Celery+Redis. В dev без брокера — запускаем в потоке,
чтобы прототип был полезен сразу, без поднятия очереди.

Модель обработки: ingest аудио — ОДИН раз на запись, затем по прогону AsrRun на каждый
выбранный распознаватель (для сравнения точности). Статус записи агрегируется из прогонов.
"""
from __future__ import annotations

import threading
import traceback

from django.conf import settings


def _aggregate(rec) -> None:
    """Свести статус записи из статусов её прогонов."""
    from .models import Status
    statuses = [r.status for r in rec.runs.all()]
    if not statuses:
        return
    if any(s == Status.READY for s in statuses):
        rec.status = Status.READY
    elif all(s == Status.ERROR for s in statuses):
        rec.status = Status.ERROR
    elif any(s == Status.PROCESSING for s in statuses):
        rec.status = Status.PROCESSING
    else:
        rec.status = Status.QUEUED
    rec.stage = ""
    rec.save(update_fields=["status", "stage", "updated_at"])


def _run_safe(run) -> None:
    """Прогнать один AsrRun, аккуратно ведя его статус/ошибку."""
    from .models import Status
    from . import pipeline
    try:
        pipeline.run_one(run)
    except Exception as e:  # noqa: BLE001
        run.status = Status.ERROR
        run.error = _friendly_error(run.stage, e)
        run.save(update_fields=["status", "error", "updated_at"])


def _run_parallel(runs) -> None:
    """Прогнать независимые прогоны параллельно в потоках (облачный google || GPU-whisper).
    Пустой/одиночный список — без потоков. Каждый поток закрывает свои Django-соединения
    (SQLite thread-local), чтобы не текли между обработками."""
    from django.db import connections
    if not runs:
        return
    if len(runs) == 1:
        _run_safe(runs[0])
        return

    def _worker(run):
        try:
            _run_safe(run)
        finally:
            connections.close_all()

    threads = [threading.Thread(target=_worker, args=(r,), daemon=True) for r in runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def run_pipeline(rec_id: int) -> None:
    """Обработать запись целиком: общий ingest + все её прогоны (queued/error)."""
    from .models import Recitation, Status
    from . import pipeline

    try:
        rec = Recitation.objects.get(pk=rec_id)
    except Recitation.DoesNotExist:
        return

    rec.status = Status.PROCESSING
    rec.error = ""
    rec.stage = "ingest"
    rec.save(update_fields=["status", "error", "stage", "updated_at"])

    # общий шаг — скачать/подготовить аудио один раз; фатальная ошибка = вся запись в error
    try:
        pipeline.ensure_audio(rec)
    except Exception as e:  # noqa: BLE001
        msg = _friendly_error("ingest", e)
        rec.status = Status.ERROR
        rec.error = msg
        rec.save(update_fields=["status", "error", "updated_at"])
        rec.runs.update(status=Status.ERROR, error=msg)
        return

    # ASR-распознаватели — ПАРАЛЛЕЛЬНО, выравниватели (forced) — строго ПОСЛЕ них.
    # google = ожидание облака (сеть), whisper = GPU-инференс: они не конкурируют за ресурс,
    # поэтому гоняем в потоках и оверлапим (раньше шли циклом — whisper висел queued, пока
    # google молотит облако; владелец 10.07: «почему виспер в это же время не начнёт?»).
    # forced монотонно после ASR — ему нужен готовый источник диапазона аятов, и он единственный
    # GPU-потребитель на своём шаге (пересечения по VRAM нет). Между записями celery --concurrency 1
    # → две записи параллельно НЕ идут (иначе 2 whisper/forced на 6ГБ GPU = OOM; масштабирование
    # через GPU-лок — в BACKLOG).
    from . import sources
    todo = list(rec.runs.filter(status__in=[Status.QUEUED, Status.ERROR]))
    # ВСЕ распознаватели равноправны и НЕЗАВИСИМЫ (владелец): гоним РОВНО выбранные в форме, никакой
    # авто-магии/«особых» источников. Не-ISOLATE (google=сеть, whisper=GPU-инференс) — ПАРАЛЛЕЛЬНО
    # (не конкурируют за ресурс). ISOLATE (w2v/forced/w2vo — каждый свой gpu_align-подпроцесс на 6ГБ) —
    # строго ПОСЛЕДОВАТЕЛЬНО (иначе два torch/onnxruntime на карте = OOM + липкая CUDA-арена).
    _run_parallel([r for r in todo if not sources.is_isolated(r.recognizer)])
    for r in (r for r in todo if sources.is_isolated(r.recognizer)):
        _run_safe(r)
    _aggregate(rec)


def run_single(run_id: int) -> None:
    """Прогнать/перегнать один распознаватель (кнопка «пересчитать/добавить распознаватель»)."""
    from .models import AsrRun
    try:
        run = AsrRun.objects.select_related("recitation").get(pk=run_id)
    except AsrRun.DoesNotExist:
        return
    # запускаем РОВНО этот распознаватель (все независимы — никаких пост-шагов поверх)
    _run_safe(run)
    _aggregate(run.recitation)


def _friendly_error(stage: str, e: Exception) -> str:
    """Человеческое сообщение для фронта (без сырого дампа yt-dlp/трейсбека).
    Полный трейс печатаем в лог сервера — на страницу его не тащим."""
    traceback.print_exc()
    msg = str(e)
    low = msg.lower()
    if "yt-dlp" in low or "youtube" in low or (stage or "").startswith("ingest"):
        return ("Не удалось скачать аудио с YouTube (анти-бот/ограничение доступа). "
                "Попробуй другую ссылку или загрузи аудиофайл напрямую. "
                "Обход через свежий yt-dlp/куки — в планах.")
    if "google stt" in low or "gstt" in low:
        return msg  # уже человекочитаемо (нет кэша и т.п.)
    return f"Ошибка обработки ({stage or '?'}): {type(e).__name__}: {msg[:200]}"


# Celery-задачи (используются, когда задан брокер)
try:
    from synchronized.celery import app as _celery_app

    @_celery_app.task(name="recitations.run_pipeline")
    def run_pipeline_task(rec_id: int):
        run_pipeline(rec_id)

    @_celery_app.task(name="recitations.run_single")
    def run_single_task(run_id: int):
        run_single(run_id)
except Exception:  # celery недоступен — не критично для dev
    run_pipeline_task = None
    run_single_task = None


def _bg(fn, arg: int) -> None:
    threading.Thread(target=fn, args=(arg,), daemon=True).start()


def dispatch(rec_id: int) -> None:
    """Поставить обработку всей записи: Celery при наличии брокера, иначе фоновый поток."""
    if settings.CELERY_BROKER_URL and run_pipeline_task is not None:
        run_pipeline_task.delay(rec_id)
    else:
        _bg(run_pipeline, rec_id)


def dispatch_run(run_id: int) -> None:
    """Поставить один прогон (пересчёт/добавление распознавателя)."""
    if settings.CELERY_BROKER_URL and run_single_task is not None:
        run_single_task.delay(run_id)
    else:
        _bg(run_single, run_id)
