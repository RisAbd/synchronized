# Ресёрч: Tarteel-AI как распознаватель (арабский Коран ASR)

> Заказ владельца (04.07): «про тартиль-ИИ не смотрел ещё? чуток подними приоритет, чтобы до
> задачки с whisper уже было понимание и потенциальный выход». Здесь — что нашёл и как это
> ложится на наш стек. Итог: **есть готовая специализированная whisper-модель под коранический
> арабский — вероятно, прямой фикс «бедного whisper», причём даже без GPU.**

## Что такое Tarteel и что они выложили

Tarteel AI — приложение-компаньон для чтения Корана с живым фидбеком по рецитации. На Hugging
Face (`huggingface.co/tarteel-ai`) выложены **модели** и **датасеты** (публично):

### Модели (ASR, whisper-семейство)
| Модель | База | Размер | WER (их eval) | Заметки |
|--------|------|--------|---------------|---------|
| `tarteel-ai/whisper-base-ar-quran` | openai/whisper-base | ~74M | **5.75%** | ⭐ 387k загрузок, Apache-2.0, декабрь 2022 |
| `tarteel-ai/whisper-tiny-ar-quran` | openai/whisper-tiny | ~39M | выше | легче, менее точная |

Оба **дообучены именно на коранической рецитации** (в отличие от ванильного whisper, который у
нас на арабском выдавал кашу — напр. rec9 на CPU дал **5 слов** на 20 мин). WER 5.75% — на их
чистом eval'е; на длинных/мелодичных читках реальность будет хуже, но всё равно кратно лучше
базового whisper.

### Комьюнити-производные (больше/точнее / с диакритикой)
- `IJyad/whisper-large-v3-Tarteel` — дообучение **large-v3** (тяжелее, лучше на сложном аудио).
- `MaddoggProduction/whisper-l-v3-turbo-quran-lora-...` — LoRA поверх **large-v3-turbo** (быстрее large).
- `KheemP/whisper-base-quran-lora` — **диакритико-чувствительная** (harakat!) LoRA, test WER ~5.98%
  → прямая связка с П5 (харакат-левел подсветка).

### Датасеты (ground truth для арабского Корана)
- `tarteel-ai/everyayah` — ⭐ **4.49k likes**, рецитации, **сегментированные по аятам**, много чтецов.
  Идеальный ground truth: (а) метрики точности наших прогонов; (б) gold word/verse-сегменты для
  популярных чтецов (см. Next п.4 QUL/quran-align); (в) дообучение при желании.
- `tarteel-ai/EA-DI` / `EA-UD` — everyayah диакритизированный / недиакритизированный (245k / 4.9k загр.).
- `tarteel-ai/tlog`, `quranqa`, `quran-tafsir` — не про ASR (логи/QA/тафсир).

Лицензия ключевой модели `whisper-base-ar-quran` — **Apache-2.0** (можно использовать/коммерция).

## Как это ложится на наш стек — «потенциальный выход»

Наш ASR-путь: `src/asr.py` → **faster-whisper** (CTranslate2), в докере CPU, на хосте GPU. Модели
Tarteel — это **обычные HF-чекпойнты whisper**. Два пути подключить:

1. **Через `transformers` pipeline** (просто, но тяжелее по deps: torch+transformers):
   ```python
   from transformers import pipeline
   pipe = pipeline("automatic-speech-recognition", model="tarteel-ai/whisper-base-ar-quran")
   ```
2. **Конвертировать в CTranslate2** и скормить нашему faster-whisper (совместимо с текущим кодом,
   быстрый инференс, CPU/GPU):
   ```bash
   ct2-transformers-converter --model tarteel-ai/whisper-base-ar-quran \
       --output_dir models/ct2-tarteel-base --quantization int8
   ```
   ```python
   WhisperModel("models/ct2-tarteel-base", device=..., compute_type="int8")
   ```

### Вывод для задачи «whisper» (п.7 разбора ТЗ) — ПЕРЕФОРМУЛИРОВАТЬ
Изначально п.7 = «whisper GPU в докере» (тяжело: nvidia-container-toolkit). **Ресёрч меняет вывод:**
корень проблемы не «CPU медленно», а **ванильная модель плохо знает коранический арабский**.
Дешёвый выход, вероятно, БЕЗ GPU:

- **Сменить модель** ванильного whisper на `tarteel-ai/whisper-base-ar-quran` (74M — CPU-дружелюбна,
  int8). Ожидаемо: rec9 перестанет давать «5 слов», покрытие/качество вырастут кратно.
- Реализация: параметр модели в `src/asr.py` (env `SYNC_WHISPER_MODEL`, дефолт можно оставить
  large-v3 на хосте, а в докере — ct2-tarteel-base int8). Один сконвертированный каталог примонтировать
  в воркер (bind-mount, как creds) — контейнер не пухнет.
- Если base не тянет длинные мелодичные читки — эскалация на `IJyad/whisper-large-v3-Tarteel` (уже GPU).
- **everyayah** — отдельно завести как источник gold-метрик (сравнивать наши google/whisper/forced
  с эталоном по WER, а не «на глаз»).

### Осторожно / ограничения
- Модель декабря 2022, карточка неполная («More information needed»), обучающий датасет в карточке
  не указан явно (вероятно everyayah).
- whisper по-прежнему **не даёт таймингов по словам напрямую** — наш `align.py`/forced align всё равно
  нужны для привязки к аятам. Tarteel улучшает ТЕКСТ распознавания, не сам alignment.
- Диакритика (harakat) — только через LoRA-вариант; для нашего forced-пути харакат и так наследуют
  тайминги базы (см. `falign.py`), так что это скорее к П5, чем к ASR.
- Для арабского у нас уже активен **google** (rec9 live coverage 0.901, rec5 1.0) — Tarteel полезен
  как **офлайн/бесплатная** альтернатива и для сравнения, но не срочный блокер прототипа.

## Источники
- https://huggingface.co/tarteel-ai — модели и датасеты org
- https://huggingface.co/tarteel-ai/whisper-base-ar-quran — WER 5.75%, Apache-2.0
- https://huggingface.co/tarteel-ai/whisper-tiny-ar-quran
- https://huggingface.co/datasets/tarteel-ai/everyayah — сегменты по аятам, много чтецов
- https://huggingface.co/IJyad/whisper-large-v3-Tarteel — large-v3 дообучение
- https://huggingface.co/KheemP/whisper-base-quran-lora — диакритико-чувствительная LoRA
