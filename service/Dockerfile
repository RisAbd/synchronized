# Образ Django-сервиса synchronized (web + worker используют один образ).
# Контекст сборки — корень репозитория (см. docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DJANGO_SETTINGS_MODULE=synchronized.settings \
    # ctranslate2 (faster-whisper GPU) ищет cuDNN/cuBLAS по LD_LIBRARY_PATH → указываем на
    # pip-колёса nvidia-*-cu12. Сам драйвер CUDA прокидывает nvidia-container-toolkit с хоста.
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib \
    # модели (whisper→HF hub, MMS-onnx→~/ctc_forced_aligner) кэшируем в примонтированный
    # /app/.cache, иначе по 3-4 ГБ докачки на каждый рестарт контейнера. HOME задаёт путь MMS.
    HF_HOME=/app/.cache/huggingface \
    HOME=/app/.cache

# ffmpeg нужен для ingest/ASR
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY service/requirements.txt /app/service/requirements.txt
# torch — CPU-колесо (forced гоняет onnxruntime; torch нужен лишь как зависимость
# ctc-forced-aligner). CUDA-торч тянуть незачем — экономит ~2 ГБ образа.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/service/requirements.txt

# код: ядро (src), данные (quran.db), веб-статика, сам сервис
COPY src /app/src
COPY data /app/data
COPY web /app/web
COPY service /app/service

WORKDIR /app/service
EXPOSE 8000

# по умолчанию — веб; worker переопределяет command в docker-compose
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn synchronized.wsgi:application -b 0.0.0.0:8000 --workers 2 --threads 4 --timeout 300"]
