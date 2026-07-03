"""Пакет проекта. Подхватываем Celery-приложение при старте Django."""
try:
    from .celery import app as celery_app  # noqa: F401
    __all__ = ("celery_app",)
except Exception:  # celery не обязателен в dev без брокера
    pass
