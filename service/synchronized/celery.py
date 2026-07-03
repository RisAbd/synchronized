"""Celery-приложение. Брокер/бэкенд — из настроек (Redis в docker-compose)."""
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "synchronized.settings")

app = Celery("synchronized")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
