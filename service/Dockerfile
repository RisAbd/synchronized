# Образ Django-сервиса synchronized (web + worker используют один образ).
# Контекст сборки — корень репозитория (см. docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DJANGO_SETTINGS_MODULE=synchronized.settings

# ffmpeg нужен для ingest/ASR
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY service/requirements.txt /app/service/requirements.txt
RUN pip install --no-cache-dir -r /app/service/requirements.txt

# код: ядро (src), данные (quran.db), веб-статика, сам сервис
COPY src /app/src
COPY data /app/data
COPY web /app/web
COPY service /app/service

WORKDIR /app/service
EXPOSE 8000

# по умолчанию — веб; worker переопределяет command в docker-compose
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn synchronized.wsgi:application -b 0.0.0.0:8000 --workers 2 --threads 4 --timeout 300"]
