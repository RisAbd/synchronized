"""Переносим legacy-данные плеера (Recitation.data) в прогоны AsrRun.

Исторически распознаватель на запись не хранился. Эвристика:
  - демо-Исра / локальные файлы — это Google STT (кэш) → recognizer='google', ставим gstt_key;
  - остальное (ссылки) обрабатывалось whisper → recognizer='whisper'.
Метрики восстанавливаем из data (кол-во точек word_timeline/timeline, длительность).
"""
from django.db import migrations


def forward(apps, schema_editor):
    Recitation = apps.get_model("recitations", "Recitation")
    AsrRun = apps.get_model("recitations", "AsrRun")

    for rec in Recitation.objects.all():
        if not rec.data:
            continue
        if rec.runs.exists():
            continue
        is_google = rec.source_type == "file" or (rec.source_url or "").startswith("local:")
        recognizer = "google" if is_google else "whisper"
        if is_google and not rec.gstt_key and "isra" in (rec.source_url or "").lower():
            rec.gstt_key = "aswailis_isra"
            rec.save(update_fields=["gstt_key"])
        wt = (rec.data or {}).get("word_timeline") or []
        tl = (rec.data or {}).get("timeline") or []
        AsrRun.objects.create(
            recitation=rec,
            recognizer=recognizer,
            status="ready",
            data=rec.data,
            metrics={"wt": len(wt), "tl": len(tl),
                     "duration": (rec.data or {}).get("duration", 0),
                     "legacy": True},
        )


def backward(apps, schema_editor):
    AsrRun = apps.get_model("recitations", "AsrRun")
    AsrRun.objects.filter(metrics__legacy=True).delete()


class Migration(migrations.Migration):
    dependencies = [
        ("recitations", "0002_recitation_gstt_key_asrrun"),
    ]
    operations = [
        migrations.RunPython(forward, backward),
    ]
