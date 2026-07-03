"""Настройки Django для сервиса synchronized.

Стандартный проект: SQLite по умолчанию (dev), Postgres через env (docker-compose),
Celery+Redis для фоновой обработки. Конфигурация — из переменных окружения.
"""
from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent      # .../service
REPO_ROOT = BASE_DIR.parent                            # .../synchronized
load_dotenv(BASE_DIR / ".env")

# --- пути к ядру пайплайна и данным (переиспользуем существующую структуру репо) ---
PIPELINE_SRC = REPO_ROOT / "src"       # quran/ingest/asr/align/player
AUDIO_DIR = REPO_ROOT / "web" / "audio"  # готовое аудио для отдачи плееру
WORK_DIR = REPO_ROOT / "work"            # промежуточные выгрузки конвейера
for _d in (AUDIO_DIR, WORK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- базовое ---
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-insecure-key-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")
# ngrok/https-прокси — доверяем заголовкам, чтобы csrf не ругался на туннель
CSRF_TRUSTED_ORIGINS = [o for o in os.environ.get(
    "DJANGO_CSRF_TRUSTED", "https://*.ngrok-free.app,https://*.ngrok.io").split(",") if o]
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "recitations",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "synchronized.urls"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.contrib.auth.context_processors.auth",
        "django.contrib.messages.context_processors.messages",
    ]},
}]
WSGI_APPLICATION = "synchronized.wsgi.application"

# --- БД: Postgres если задан POSTGRES_DB, иначе SQLite ---
if os.environ.get("POSTGRES_DB"):
    DATABASES = {"default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ["POSTGRES_DB"],
        "USER": os.environ.get("POSTGRES_USER", "postgres"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "postgres"),
        "HOST": os.environ.get("POSTGRES_HOST", "db"),
        "PORT": os.environ.get("POSTGRES_PORT", "5432"),
    }}
else:
    DATABASES = {"default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }}

AUTH_PASSWORD_VALIDATORS = []
LANGUAGE_CODE = "ru-ru"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Celery (фоновая обработка). Если брокер не задан — задачи гоняем в потоке (dev). ---
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", CELERY_BROKER_URL or "cache+memory://")
CELERY_TASK_ALWAYS_EAGER = not CELERY_BROKER_URL and os.environ.get("CELERY_EAGER", "0") == "1"

# --- распознаватель речи: whisper (локально) | google (кэш/API) ---
# ⚠️ ключ Google НЕ хранить в репо — только путь через GOOGLE_APPLICATION_CREDENTIALS.
RECOGNIZER = os.environ.get("SYNC_RECOGNIZER", "whisper")
GSTT_CACHE_DIR = os.environ.get(
    "GSTT_CACHE_DIR",
    str(REPO_ROOT.parent / "speech-to-text-python" / "gcloud-speech-data"))
