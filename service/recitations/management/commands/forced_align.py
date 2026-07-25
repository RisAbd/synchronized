"""Прогнать forced align (src/falign.py) для записи ВНЕ докера — на хосте, где стоят
зависимости (ctc-forced-aligner, onnxruntime, unidecode).

Зачем отдельно: пайплайн в docker-воркере крутится на CPU-slim образе без этих пакетов
(как и whisper-GPU). Пока их нет в контейнере — forced-прогон генерим здесь, на хосте;
БД (SQLite) и media примонтированы bind-mount, поэтому web-контейнер сразу отдаёт результат.

forced теперь ПОЛНОСТЬЮ независим (инкремент 3): сам находит диапазон из своей акустики MMS,
готовый ASR-прогон больше не нужен.

    python manage.py forced_align 5           # одна запись
    python manage.py forced_align 5 6 7        # несколько
    python manage.py forced_align --all        # все записи
"""
from __future__ import annotations

import time

from django.core.management.base import BaseCommand

from recitations import pipeline
from recitations.models import AsrRun, Recitation


class Command(BaseCommand):
    help = "Forced alignment по известному тексту для записи (на хосте, вне докера)."

    def add_arguments(self, parser):
        parser.add_argument("rec_ids", nargs="*", type=int, help="id записей")
        parser.add_argument("--all", action="store_true", help="все записи")

    def handle(self, *args, **opts):
        ids = list(opts["rec_ids"])
        if opts["all"]:
            for rec in Recitation.objects.all():
                if rec.id not in ids:
                    ids.append(rec.id)
        if not ids:
            self.stderr.write("нечего делать: укажи id записей или --all-ready")
            return

        for rec_id in ids:
            try:
                rec = Recitation.objects.get(pk=rec_id)
            except Recitation.DoesNotExist:
                self.stderr.write(f"[{rec_id}] нет такой записи")
                continue
            run, _ = AsrRun.objects.get_or_create(recitation=rec, recognizer="forced")
            run.status = AsrRun.Status.QUEUED
            run.error = ""
            run.save(update_fields=["status", "error", "updated_at"])
            self.stdout.write(f"[{rec_id}] forced align…")
            t0 = time.monotonic()
            try:
                pipeline.run_one(run, on_stage=lambda s: self.stdout.write(f"  … {s}"))
            except Exception as e:  # noqa: BLE001
                run.status = AsrRun.Status.ERROR
                run.error = str(e)
                run.save(update_fields=["status", "error", "updated_at"])
                self.stderr.write(f"[{rec_id}] ошибка: {e}")
                continue
            m = run.metrics or {}
            self.stdout.write(self.style.SUCCESS(
                f"[{rec_id}] готово за {time.monotonic()-t0:.0f}с: "
                f"слов {m.get('wt')}, символов {m.get('ct')}, coverage {m.get('coverage')}"))
