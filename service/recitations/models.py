"""Модели записи чтения (recitation) и её ASR-прогонов (по распознавателям)."""
import re

from django.db import models

from . import recognizers

# id видео из youtube.com/watch?v=…, youtu.be/…, youtube.com/embed/… (11 символов)
_YT_ID_RE = re.compile(r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/)|youtu\.be/)([\w-]{11})")


class Status(models.TextChoices):
    QUEUED = "queued", "в очереди"
    PROCESSING = "processing", "обрабатывается"
    READY = "ready", "готово"
    ERROR = "error", "ошибка"


class Recitation(models.Model):
    Status = Status  # обратная совместимость: Recitation.Status.READY и т.п.

    source_url = models.CharField("источник (ссылка/файл)", max_length=1000)
    source_type = models.CharField(max_length=16, default="youtube")  # youtube|file|other

    title = models.CharField(max_length=300, blank=True)
    title_ar = models.CharField(max_length=300, blank=True)
    reciter = models.CharField(max_length=300, blank=True)

    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    stage = models.CharField(max_length=64, blank=True)   # текущий шаг конвейера (для прогресса)
    error = models.TextField(blank=True)

    audio_filename = models.CharField(max_length=300, blank=True)  # имя файла аудио
    # Ключ подпапки кэша Google STT (если отличается от stem аудио), напр. "aswailis_isra".
    gstt_key = models.CharField(max_length=200, blank=True)
    # Данные плеера для legacy-записей (до модели AsrRun). Новые записи держат data в AsrRun.
    data = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.title or self.source_url} [{self.status}]"

    @property
    def youtube_id(self):
        """id YouTube-видео из source_url (для эмбеда рядом с текстом, П1). '' если не YouTube."""
        if self.source_type != "youtube":
            return ""
        m = _YT_ID_RE.search(self.source_url or "")
        return m.group(1) if m else ""

    # --- прогоны по распознавателям ---

    def ready_runs(self):
        return [r for r in self.runs.all() if r.status == Status.READY]

    def active_run(self, prefer: str | None = None):
        """Активный прогон для плеера: явно запрошенный (если готов), иначе первый готовый
        по приоритету распознавателей."""
        ready = {r.recognizer: r for r in self.ready_runs()}
        if prefer and prefer in ready:
            return ready[prefer]
        for key in recognizers.PRIORITY:
            if key in ready:
                return ready[key]
        return next(iter(ready.values()), None)

    @property
    def is_ready(self):
        # запись «готова», если готов хотя бы один прогон (или есть legacy-data)
        if self.status == Status.READY:
            return True
        return any(r.status == Status.READY for r in self.runs.all())

    def _player_data(self):
        run = self.active_run()
        if run and run.data:
            return run.data
        return self.data or {}

    @property
    def duration(self):
        return self._player_data().get("duration", 0)

    @property
    def surahs_label(self):
        secs = self._player_data().get("sections", [])
        return " · ".join(f"سورة {s['title']}" for s in secs)


class AsrRun(models.Model):
    """Один прогон конвейера конкретным распознавателем поверх общего аудио записи.
    Несколько прогонов на одну запись = сравнение точности разных ASR."""
    Status = Status

    recitation = models.ForeignKey(Recitation, related_name="runs", on_delete=models.CASCADE)
    recognizer = models.CharField(max_length=32)   # ключ из recognizers.REGISTRY

    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    stage = models.CharField(max_length=64, blank=True)
    error = models.TextField(blank=True)

    data = models.JSONField(null=True, blank=True)      # выход build_data (для плеера)
    metrics = models.JSONField(null=True, blank=True)   # покрытие/слова/сегменты/тайминг

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("recognizer",)
        constraints = [
            models.UniqueConstraint(fields=["recitation", "recognizer"],
                                    name="uniq_recitation_recognizer"),
        ]

    def __str__(self):
        return f"{self.recitation_id}/{self.recognizer} [{self.status}]"

    @property
    def is_ready(self):
        return self.status == Status.READY

    @property
    def label(self):
        return recognizers.label_of(self.recognizer)
