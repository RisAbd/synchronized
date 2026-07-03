"""Сид демо-записи из готовых данных (web/data/aswailis-isra.json), чтобы библиотека была не пустой."""
import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from recitations.models import AsrRun, Recitation


class Command(BaseCommand):
    help = "Создать демо-запись (Аль-Исра) из web/data/aswailis-isra.json, если её ещё нет."

    def handle(self, *args, **opts):
        data_path = settings.REPO_ROOT / "web" / "data" / "aswailis-isra.json"
        if not data_path.is_file():
            self.stderr.write(f"нет файла: {data_path}")
            return
        data = json.loads(data_path.read_text())
        audio_name = data.get("audio", "aswailis-isra.mp3")
        if not (Path(settings.AUDIO_DIR) / audio_name).is_file():
            self.stderr.write(f"нет аудио: {settings.AUDIO_DIR}/{audio_name} (положи web/audio/{audio_name})")

        title = "Сура Аль-Исра (фрагмент)"
        if Recitation.objects.filter(title=title).exists():
            self.stdout.write("демо уже есть — пропускаю")
            return
        rec = Recitation.objects.create(
            source_url="local:aswailis-isra",
            source_type="file",
            title=title,
            title_ar=data.get("title_ar", "سورة الإسراء"),
            reciter=data.get("reciter", "الشيخ يونس اسويلص"),
            gstt_key="aswailis_isra",
            status=Recitation.Status.READY,
            audio_filename=audio_name,
            data=data,
        )
        wt = data.get("word_timeline") or []
        tl = data.get("timeline") or []
        AsrRun.objects.create(
            recitation=rec, recognizer="google", status=AsrRun.Status.READY, data=data,
            metrics={"wt": len(wt), "tl": len(tl), "duration": data.get("duration", 0), "legacy": True},
        )
        self.stdout.write(self.style.SUCCESS("демо-запись создана"))
