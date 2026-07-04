"""Модели записи чтения (recitation) и её ASR-прогонов (по распознавателям)."""
import re
import unicodedata

from django.db import models


def _ar_norm(s: str) -> str:
    """Нормализация для поиска: убрать харакат (combining marks) и унифицировать алифы/я/та-марбуту,
    привести к нижнему регистру. Чтобы «الأنعام» находило хранимое «ٱلْأَنْعَام»."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    for a in "آأإٱٲٳ":
        s = s.replace(a, "ا")
    return s.replace("ى", "ي").replace("ة", "ه").lower()

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
    # Метаинфо источника (П6): {duration, filesize, ext, thumbnail, yt_title}. Заполняется при ingest.
    meta = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.title or self.source_url} [{self.status}]"

    @classmethod
    def find_by_source(cls, url: str):
        """Найти уже существующую запись по источнику (П7: не запускать повторно).
        Для YouTube сверяем по id видео (разные формы ссылки → одно видео), иначе по точной ссылке."""
        url = (url or "").strip()
        if not url:
            return None
        m = _YT_ID_RE.search(url)
        if m:
            yid = m.group(1)
            for r in cls.objects.filter(source_type="youtube"):
                if r.youtube_id == yid:
                    return r
            return None
        return cls.objects.filter(source_url=url).first()

    def matches_query(self, q: str) -> bool:
        """Совпадает ли запись с поисковым запросом (по названию/чтецу/ссылке/названиям сур).
        Арабский нормализуем (без харакат/варианты алифа), чтобы «الأنعام» находило «ٱلْأَنْعَام»."""
        if not (q or "").strip():
            return True
        hay = " ".join([self.title or "", self.reciter or "", self.source_url or "",
                        self.title_ar or "", self.surahs_label])
        return _ar_norm(q) in _ar_norm(hay)

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

    @property
    def has_active_runs(self):
        """Есть ли прогоны в работе/очереди — карточку надо опрашивать, даже если запись
        «готова» по старым прогонам (пересчёт: чипы прогонов обновляются вживую)."""
        return any(r.status in (Status.QUEUED, Status.PROCESSING) for r in self.runs.all())

    def _player_data(self):
        run = self.active_run()
        if run and run.data:
            return run.data
        return self.data or {}

    @property
    def duration(self):
        # длительность из плеера (после align) или из метаинфо аудио (доступна до готовности)
        return self._player_data().get("duration") or (self.meta or {}).get("duration") or 0

    @property
    def duration_h(self):
        """Длительность в формате m:ss (или h:mm:ss) — для списка/детализации."""
        s = int(self.duration or 0)
        if not s:
            return ""
        h, m, sec = s // 3600, (s % 3600) // 60, s % 60
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

    @property
    def filesize_h(self):
        """Размер файла человекочитаемо (МБ)."""
        b = (self.meta or {}).get("filesize") or 0
        if not b:
            return ""
        mb = b / (1024 * 1024)
        return f"{mb:.1f} МБ" if mb >= 0.1 else f"{b // 1024} КБ"

    @property
    def thumbnail(self):
        """URL превью (для YouTube — кадр по id; для файлов пусто)."""
        t = (self.meta or {}).get("thumbnail")
        if t:
            return t
        yid = self.youtube_id
        return f"https://img.youtube.com/vi/{yid}/hqdefault.jpg" if yid else ""

    @property
    def ext_h(self):
        return ((self.meta or {}).get("ext") or "").lstrip(".").upper()

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

    @property
    def is_aligner(self):
        """forced align — не распознаватель, а выравниватель известного текста.
        В UI показываем отдельно от «распознавание:» (см. player.html)."""
        return recognizers.is_aligner(self.recognizer)
