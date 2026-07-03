"""Фоновая обработка записи: Celery-задача + диспетчер.

В docker-compose работает через Celery+Redis. В dev без брокера — запускаем в потоке,
чтобы прототип был полезен сразу, без поднятия очереди.
"""
from __future__ import annotations

import threading
import traceback

from django.conf import settings


def run_pipeline(rec_id: int) -> None:
    """Прогнать конвейер для записи, аккуратно ведя статус/ошибку."""
    from .models import Recitation
    from . import pipeline

    try:
        rec = Recitation.objects.get(pk=rec_id)
    except Recitation.DoesNotExist:
        return

    rec.status = Recitation.Status.PROCESSING
    rec.error = ""
    rec.save(update_fields=["status", "error", "updated_at"])
    try:
        pipeline.process(rec)
        rec.status = Recitation.Status.READY
        rec.stage = ""
        rec.save(update_fields=["status", "stage", "updated_at"])
    except Exception as e:  # noqa: BLE001
        rec.status = Recitation.Status.ERROR
        rec.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        rec.save(update_fields=["status", "error", "updated_at"])


# Celery-задача (используется, когда задан брокер)
try:
    from synchronized.celery import app as _celery_app

    @_celery_app.task(name="recitations.run_pipeline")
    def run_pipeline_task(rec_id: int):
        run_pipeline(rec_id)
except Exception:  # celery недоступен — не критично для dev
    run_pipeline_task = None


def dispatch(rec_id: int) -> None:
    """Поставить обработку: Celery при наличии брокера, иначе фоновый поток."""
    if settings.CELERY_BROKER_URL and run_pipeline_task is not None:
        run_pipeline_task.delay(rec_id)
    else:
        t = threading.Thread(target=run_pipeline, args=(rec_id,), daemon=True)
        t.start()
