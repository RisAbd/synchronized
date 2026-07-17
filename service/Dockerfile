# Образ Django-сервиса synchronized (web + worker используют один образ).
# Контекст сборки — корень репозитория (см. docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DJANGO_SETTINGS_MODULE=synchronized.settings \
    # ctranslate2 (faster-whisper GPU) и onnxruntime-gpu (forced align CUDA EP) ищут CUDA-либы
    # по LD_LIBRARY_PATH → указываем на pip-колёса nvidia-*-cu12 (cuDNN/cuBLAS для ct2; плюс
    # cuda_runtime/cufft/curand/nvrtc/nvjitlink для onnxruntime CUDA EP). Драйвер CUDA прокидывает
    # nvidia-container-toolkit с хоста. При сборке от root все колёса лежат в /usr/local (не .local).
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/site-packages/nvidia/nvjitlink/lib \
    # модели (whisper→HF hub, MMS-onnx→~/ctc_forced_aligner) кэшируем в примонтированный
    # /app/.cache, иначе по 3-4 ГБ докачки на каждый рестарт контейнера. HOME задаёт путь MMS.
    HF_HOME=/app/.cache/huggingface \
    HOME=/app/.cache \
    # nltk (whisperx.alignment делит текст на предложения) лениво качает punkt_tab при первом
    # align → кладём в примонтированный ./cache, чтобы не докачивать на рестарт контейнера.
    NLTK_DATA=/app/.cache/nltk_data

# ffmpeg — для ingest/ASR; g++ — ctc-forced-aligner собирает C++-расширение из исходников
# (иначе `command 'g++' failed: No such file or directory` при pip install).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY service/requirements.txt /app/service/requirements.txt
# torch — CUDA-колесо cu124 (2.5.1). Оно приносит ВЕСЬ CUDA 12.4-стек одним непротиворечивым
# набором: cuBLAS 12.4.5.8 + cuDNN 9.1.0.70 (как раньше пинилось руками) + runtime/cufft/curand/
# nvrtc/nvjitlink 12.4.127. Этот стек обслуживает сразу: (а) сам torch — wav2vec2-выравниватель
# whisperx (src/w2v_align.py); (б) onnxruntime-gpu 1.20.1 — MMS forced (SONAME .12/.11/.10 те же);
# (в) ctranslate2 — faster-whisper. Раньше torch был CPU-колесом, а nvidia-* пинились в
# requirements.txt (частью на 12.9) — при добавлении w2v это конфликтовало, поэтому стек
# унифицирован на cu124 (пины из requirements убраны, см. комментарий там). Ставим torch ПЕРВЫМ,
# чтобы он зафиксировал версии nvidia-*, а requirements их уже не трогал.
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r /app/service/requirements.txt \
    # whisperx: ставим БЕЗ его зависимостей (--no-deps) — из всего пакета нам нужен лишь путь
    # выравнивания (whisperx.align/load_align_model/load_audio, см. src/w2v_align.py), который тянет
    # только torch/torchaudio/transformers/pandas/nltk. Полный dep-хвост whisperx (faster-whisper,
    # pyannote и onnxruntime-CPU) не нужен и КОНФЛИКТУЕТ с onnxruntime-gpu (обе кладут модуль
    # `onnxruntime`). transformers пинним под torch 2.5 (jonatasgrosman wav2vec2-large-xlsr-53-arabic).
    && pip install --no-cache-dir --no-deps whisperx==3.8.6 \
    && pip install --no-cache-dir "transformers==4.46.3" pandas nltk

# HOME=/app/.cache (примонтированный ./cache) → Python иначе добавляет $HOME/.local/.../site-packages
# в путь ПЕРЕД образом; там от старых экспериментов лежал user-site transformers 5.x, затенявший
# наш /usr/local 4.46.3 и ломавший whisperx (ImportError huggingface_hub). Игнорируем user-site —
# авторитетны только пакеты образа. Позднее ENV (после pip) → не инвалидирует дорогой pip-слой.
ENV PYTHONNOUSERSITE=1

# код: ядро (src), данные (quran.db), веб-статика, сам сервис
COPY src /app/src
COPY data /app/data
COPY web /app/web
COPY service /app/service

WORKDIR /app/service
EXPOSE 8000

# по умолчанию — веб; worker переопределяет command в docker-compose
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn synchronized.wsgi:application -b 0.0.0.0:8000 --workers 2 --threads 4 --timeout 300"]
