"""Модель записи чтения (recitation) и её статусов обработки."""
from django.db import models


class Recitation(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "в очереди"
        PROCESSING = "processing", "обрабатывается"
        READY = "ready", "готово"
        ERROR = "error", "ошибка"

    source_url = models.CharField("источник (ссылка/файл)", max_length=1000)
    source_type = models.CharField(max_length=16, default="youtube")  # youtube|file|other

    title = models.CharField(max_length=300, blank=True)
    title_ar = models.CharField(max_length=300, blank=True)
    reciter = models.CharField(max_length=300, blank=True)

    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    stage = models.CharField(max_length=64, blank=True)   # текущий шаг конвейера (для прогресса)
    error = models.TextField(blank=True)

    audio_filename = models.CharField(max_length=300, blank=True)  # имя в AUDIO_DIR
    # данные для плеера: {audio, timeline, sections, chapters, duration}
    data = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.title or self.source_url} [{self.status}]"

    @property
    def is_ready(self):
        return self.status == self.Status.READY

    @property
    def duration(self):
        return (self.data or {}).get("duration", 0)

    @property
    def surahs_label(self):
        secs = (self.data or {}).get("sections", [])
        return " · ".join(f"سورة {s['title']}" for s in secs)
