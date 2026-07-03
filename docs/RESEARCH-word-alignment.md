# Исследование: пословная (и до-хараке) привязка аудио ↔ текст

Дата: 2026-07-04. Триггер — владелец: «не аят в центре, а слово надо подсвечивать и плавно
крутить… хотел вплоть до хареке! но не вышло даже по словам, poor выходило. Надо изучать».

## Проблема (где застряли)

Нужно сопоставить каждое слово канона с интервалом аудио, чтобы подсвечивать слово и плавно
скроллить. Текущий путь (whisper `word_timestamps` → `align.py` на канон) по словам даёт poor:

- **Складываются два источника ошибки:** (1) whisper-тайминги слов неточны (attention/DTW-
  эвристика, ±150–300 мс), (2) маппинг ASR→канон лоссовый (покрытие 70–90%, часть слов без
  тайминга, повторы/нормализация путают).
- **Мадд и вакф.** Whisper выдаёт слово почти «точкой», а чтец тянет слово (мадд) секундами и
  делает паузы (вакф). Интервалы разъезжаются — ровно то, что наблюдал владелец.

## Ключевой реворк: forced alignment по ИЗВЕСТНОМУ тексту

Мы **знаем** канонический текст — в этом вся идея продукта. Значит таймингом должен заниматься
не ASR («угадай слова»), а **forced alignment**: дано (текст + аудио) → оптимальная привязка
каждого юнита ко времени. Фрейм-левел Витерби назначает каждый ~20 мс фрейм токену или blank:

- **мадд** → много фреймов на одну фонему (растяжение ловится естественно);
- **пауза/вакф** → blank-фреймы (пауза не сдвигает слова).

Это прямое лекарство от «фигни в таймстемпах». `align.py` при этом **не исчезает**: он
**локализует место** (сура:аят, смена суры) — это по-прежнему нужно, особенно для live и
произвольных ссылок. Forced alignment работает поверх уже найденного места и даёт точные
границы слов.

## Готовые данные (снимают проблему для популярных чтецов)

- **QUL (Quranic Universal Library, qul.tarteel.ai) → «With Segments».** Для 50+ чтецов
  (AbdulBaset AbdulSamad, Sudais, Yasser Al-Dosari, Mishary Alafasy, Husary…) есть пословные
  сегменты. Формат: на слово `[word_number, start_ms, end_ms]`; для сур — плюс `timestamp_from/to`
  (границы аятов) и `duration_ms`. Скачивание с сайта; API «coming soon».
- **cpfair/quran-align** (то, что использовал quran.com). Метод: PocketSphinx (CMUSphinx)
  распознавание по Qur'an-specific модели + рефайнмент по MFCC до слоговых границ. Точность:
  среднее отклонение **<73 мс**, 98.5–99.9% слов сегментированы. Вход — аудио EveryAyah + текст,
  выход — JSON `{surah, ayah, segments[...]}`.
- **lafzize** (~rehandaphedar) — отдельный генератор пословных таймстемпов.

Вывод: для мейнстрим-чтецов пословную подсветку можно взять **готовой (gold)** и заодно
использовать как **ground truth** для оценки нашего аллайнера. НО обычный кейс владельца —
**obscure чтец** (يونس اسويلص, aswailis) с YouTube + будущий **live** → там готовых данных нет,
нужен свой аллайнер. Поэтому свой forced-aligner — ядро, gold — быстрый путь для популярных +
эталон для метрик.

## Свой аллайнер: инструменты

- **`MahmoudAshraf97/ctc-forced-aligner`** — обёртка над CTC forced alignment; Arabic +
  гранулярность sentence/word/**char** (char → путь к хараке). Рекомендуемый быстрый старт.
- **`torchaudio.functional.forced_align` / `Wav2Vec2FABundle`** — базовый API. ⚠️ deprecated в
  2.8, убирают в 2.9 — брать актуальную обёртку/версию.
- **Quran-специфичные wav2vec2 CTC модели (HuggingFace):**
  - `TBOGamer22/wav2vec2-quran-phonetics` — фонетическая транскрипция рецитации (sound-level),
    первый публичный wav2vec2 под Qur'an phonetics → под харакат/фонемы.
  - `IbrahimSalah/Wav2vecLarge_quran_syllables_recognition` — слоговое распознавание.
  - Tarteel ASR (whisper, дообучен на рецитации; `yazinsai/offline-tarteel`) — для **локализации**
    (surah/ayah по аудио).

Quran-специфичная CTC-акустика **критична**: обычные MSA-модели плохи на таджвиде.

## Харакат-левел (стадия 2, амбициозно)

Нужна фонемная/символьная привязка + маппинг фонема→диакритика в известном тексте (G2P
коранической орфографии — сложно, но текст фиксирован). Есть профильная работа «Improving
Automatic Forced Alignment for Phoneme Segmentation in Quranic Recitation» (учёт правил
элонгации/таджвида). MFA (Kaldi) с Arabic-моделями даёт фонемные границы, но таджвид/мадд
отличается от MSA — нужны рефайнменты. Порядок: **сначала надёжный word-level, потом
phoneme→harakat**.

## План эксперимента (spike)

1. Клип Аль-Исра (`work/audio.mp3`) + известный текст суры (есть из `align`).
2. Прогнать `ctc-forced-aligner` с Quran wav2vec2 (char+word granularity) → пословные границы.
3. Сравнить с whisper-таймингами и (если достанем) с QUL/quran-align gold по мейнстрим-чтецу.
4. Метрика: среднее |Δ| границ слова vs. ручная проверка нескольких слов; глазами — попадает ли
   подсветка на мадд/паузы.
5. Если ок → заменить источник таймингов в sync-map (align.py остаётся для локализации).

## Влияние на контракт

sync-map получает `word_timeline`: `[{t_start, t_end, surah, ayah, word_index, corpus}]`.
Плеер рендерит каждый токен аята отдельным span'ом (`id = surah:ayah:wi`), подсвечивает текущий,
плавно центрирует. Позже — sub-word (harakat) слой.

## Источники

- [QUL — With Segments](https://qul.tarteel.ai/docs/with-segments), [QUL](https://qul.tarteel.ai/)
- [cpfair/quran-align](https://github.com/cpfair/quran-align)
- [lafzize](https://sr.ht/~rehandaphedar/lafzize/)
- [torchaudio CTC forced alignment](https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html)
- [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- [TBOGamer22/wav2vec2-quran-phonetics](https://huggingface.co/TBOGamer22/wav2vec2-quran-phonetics)
- [IbrahimSalah/Wav2vecLarge_quran_syllables_recognition](https://huggingface.co/IbrahimSalah/Wav2vecLarge_quran_syllables_recognition)
- [offline-tarteel](https://github.com/yazinsai/offline-tarteel)
- [Improving Automatic Forced Alignment for Phoneme Segmentation in Quranic Recitation](https://www.researchgate.net/publication/376765478)
