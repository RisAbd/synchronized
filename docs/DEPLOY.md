# Развёртка с нуля (synchronized)

> Что нужно, чтобы поднять сервис на чистой машине (или после потери gitignored-данных).
> Всё, что НЕ в git: БД `service/db.sqlite3`, `creds/`, `cache/` (модели), `work/`, `.env`.
> Этот файл — чтобы разворачивать быстро и повторяемо (а не по кусочкам вслепую).

## 0. Предпосылки (хост)

- Docker + Docker Compose.
- **NVIDIA GPU + `nvidia-container-toolkit`** (whisper/forced гоняются на CUDA). Установка —
  см. шапку `docker-compose.yml`. Проверка: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.
- Контейнеры пишут файлы под **uid 1000** (`user: "1000:1000"` в compose) — папки-монтирования
  должны принадлежать uid 1000 (см. §3).

## 1. Секреты

### Google STT (`creds/gstt.json`)
Ключ сервис-аккаунта Google (Speech-to-Text + Cloud Storage). Лежит в проекте
`../speech-to-text-python/` (`pacific-vault-142810-*.json`). Скопировать:
```bash
mkdir -p creds
cp ../speech-to-text-python/pacific-vault-142810-*.json creds/gstt.json
```
Путь в контейнер пробрасывается как `GOOGLE_APPLICATION_CREDENTIALS=/app/creds/gstt.json`.

### `.env` (в репо не коммитится)
```
SYNC_GSTT_BUCKET=pacific-vault-142810.appspot.com
```
Бакет нужен для «живого» Google STT (аудио >1 мин заливается в GCS и распознаётся
`long_running_recognize`). Без бакета google тихо отключается, запись живёт по whisper.
НЕ класть сюда `SYNC_WHISPER_MODEL=large-v3` — дефолт (tarteel) точнее на арабском.

## 2. Модели → `cache/` (gitignored, ~1 ГБ)

### Whisper: tarteel-ai/whisper-base-ar-quran → CTranslate2
Ванильный large-v3 на арабском ПЛОХ (rec9: 5 слов на 21 мин) — нужен именно tarteel.
Конверсию делаем **в контейнере** (на хосте torchaudio тянет несовместимый CUDA13):
```bash
docker compose exec worker sh -c '
  export HF_HOME=/app/.cache/hf
  pip install -q transformers   # в образе нет; одноразово, результат осядет в ./cache
  D=$(python -c "from huggingface_hub import snapshot_download as d; print(d(\"tarteel-ai/whisper-base-ar-quran\"))")
  python -c "from transformers import WhisperTokenizerFast as T; t=T.from_pretrained(\"'$D'\"); t.save_pretrained(\"'$D'\")"  # генерит tokenizer.json (в репо его нет)
  ct2-transformers-converter --model "$D" --output_dir /app/.cache/ct2-tarteel-base \
    --copy_files tokenizer.json preprocessor_config.json --quantization float16
'
```
Итог: `cache/ct2-tarteel-base/{model.bin,config.json,tokenizer.json,vocabulary.json,preprocessor_config.json}`.
`SYNC_WHISPER_MODEL` (compose default) уже указывает на `/app/.cache/ct2-tarteel-base`.

### Forced aligner (MMS ONNX)
Тянется САМ при первом forced-прогоне в `cache/ctc_forced_aligner/model.onnx` (~800 МБ,
`falign._session_and_tokenizer` → `cfa.ensure_onnx_model`). Если файл битый/обрезан
(`INVALID_PROTOBUF`) — удалить и перезапустить прогон, перекачается.

## 3. Владелец папок-монтирований (частый грабли)

Docker при старте создаёт ОТСУТСТВУЮЩИЕ папки-монтирования как **root** → контейнер (uid 1000)
не может писать (`Permission denied … /app/work`). Починка без хостового sudo — из контейнера:
```bash
docker compose exec -u 0 worker chown -R 1000:1000 /app/work /app/.cache
```
(проверить: `ls -land work cache` → uid 1000).

## 4. Подъём

```bash
docker compose up -d --build          # соберёт web+worker (GPU-образ), поднимет redis
docker compose exec worker python manage.py migrate   # создаст service/db.sqlite3
```
Проверка GPU в воркере: `docker compose exec worker python -c \
  "import ctranslate2 as c, onnxruntime as o; print('ct2 cuda:', c.get_cuda_device_count(), o.get_available_providers())"`
(`torch` в образе — CPU-сборка НАМЕРЕННО: whisper идёт через ct2, forced через onnxruntime-gpu).

## 5. Наполнение записями (recitation = «рек»)

Только по ссылке — диапазон аятов пайплайн определяет сам:
```bash
curl -s localhost:8000/add -d 'source_url=https://youtu.be/<id>&title=...&reciter=...'
```
Пайплайн: скачать (yt-dlp, дефолт-клиент — принудительные android/ios/web YouTube сломал) →
ASR `whisper`‖`google` параллельно → `forced` авто-пост-шаг. `SYNC_RECOGNIZERS`
(compose default `whisper,google`) — какие ASR по умолчанию.

Массовый ре-импорт из выгрузки `recitations.json` — см. `tools/build_ghio.py` (обратный сценарий:
из бэка В выгрузку) и историю в CLAUDE.md.

## 6. Статичная выгрузка (GitHub Pages)

Выкладываем плеер на `risabd.github.io/syncronized/` как сабмодуль. **Всё вручную** (не
автоматизируем — владелец: «на каждый чих не надо»). **Docker поднимаю Я сам — владелец не должен;**
он только говорит «выгрузи на гитхаб.ио». Шаги:

### (0) Поднять docker из АКТУАЛЬНОГО рабочего кода (это на мне, не на владельце)
```bash
git checkout main && git pull                 # последняя рабочая версия
docker compose up -d worker redis             # веб не нужен (порт 8000 занят чужим)
docker compose exec -T worker python manage.py migrate --noinput
docker compose exec worker python manage.py shell -c "from recitations.models import Recitation as R; print('recs:', [r.id for r in R.objects.order_by('id')])"
```
Если БД/модели/креды потеряны — сперва §1–§4 (развёртка с нуля), потом сюда.

### (1) Собрать выгрузку — внутри docker, БЕЗ веба (порт 8000 занят чужим проектом)
```bash
docker compose exec -T worker python manage.py shell < tools/build_ghio.py
```
Гонит те же вьюхи, что живой бэк (RequestFactory → паритет), пишет в `./work/ghio-export/`:
`recitations.json` + `r/<id>/data.json` (forced по умолчанию, `audio=""`, manual-прогоны
выкинуты) + `index.html`/`player.html` (статика с относит. путями `./`). Только ready-записи
с youtube_id — источник видео, mp3 не выгружаем.

### (2) Положить на ветку github.io (worktree — НЕ трогать занятой main!)
```bash
WT=../synchronized.worktrees/github-io
git worktree add "$WT" github.io          # ветка-orphan уже есть; если нет — add -b github.io "$WT"
rsync -a --delete --exclude='.git' work/ghio-export/ "$WT"/
cd "$WT" && git add -A && git commit -m "github.io: обновить выгрузку" && git push
```

### (3) Обновить сабмодуль в risabd.github.io (разово add, дальше — update)
```bash
cd ../risabd.github.io
# ПЕРВЫЙ раз: git submodule add -b github.io https://github.com/RisAbd/synchronized.git syncronized
git submodule update --remote syncronized   # подтянуть свежую вершину github.io
git add syncronized && git commit -m "bump syncronized" && git push
```
Pages пересобирается ~1–2 мин (клонирует сабмодуль по https — репо публичный). Проверка:
`curl -sI https://risabd.github.io/syncronized/recitations.json` → 200. Плеер грузит
`./r/<id>/data.json` относительно документа, поэтому работает из подпапки Pages без правок.
