"""Forced alignment по ИЗВЕСТНОМУ тексту (CTC, MMS multilingual через ctc-forced-aligner).

В отличие от `align.py` (выравнивает РАСПОЗНАННЫЙ ASR-текст к корпусу и потому наследует
ошибки ASR — путает похожие слова, сажает ложные ранние якоря), здесь мы выравниваем
ground-truth текст аятов из `quran.db` прямо к аудио. ASR-ошибок нет в принципе → границы
слов/аятов точные (в т.ч. чинит баг ложного якоря на похожих словах, напр. Аль-Фуркан 25:65→66).

Вход — диапазон аятов (его берём из уже готового прогона google/whisper, где align.py определил,
что вообще читается). Выход — sync_map, СОВМЕСТИМЫЙ с `player.build_data`:
    {meta, timeline:[{t,surah,ayah}], word_timeline:[{t,surah,ayah,wi}]}

Замечания:
  * onnxruntime в этом окружении — CPU-билд (нет CUDAExecutionProvider); ~0.3× RT на CPU.
  * Романизация (uroman/unidecode) схлопывает огласовки, поэтому истинного ПО-ХАРАКАТ тайминга
    MMS не даёт. Посимвольную дорожку (`char_timeline`) строим ИНТЕРПОЛЯЦИЕЙ внутри уже точных
    границ слова — этого достаточно для плавной побуквенной подсветки в плеере. Истинный
    посимвольный CTC-тайминг — отдельная задача (см. docs/BACKLOG.md).

Требует env `LD_LIBRARY_PATH` на pip-cuDNN только если позовут CUDA; на CPU не нужен.
Зависимости: ctc-forced-aligner, onnxruntime, unidecode (uroman fallback).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# combining-марки (харакат/шадда/сукун/танвин/мадда и т.п.) — «висят» на предыдущей букве
_HARAKAT = set("ًٌٍَُِّْٰٕٖٓٔٗ٘")


# правило плотности из quran-align (segment.cc, MIN_WORD_LEN=100мс): слово короче этого —
# физически невозможно; несколько подряд — CTC «утрамбовал» текст, которого нет в аудио
MIN_WORD_SEC = 0.100
STUFFED_RUN = 3     # столько «мгновенных» слов подряд считаем скомканным участком


def _stuffed_runs(bounds: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Скомканные участки как непрерывные диапазоны [i0, i1] индексов слов.

    Правило плотности из quran-align (segment.cc, MIN_WORD_LEN=100мс): ≥ STUFFED_RUN подряд
    слов короче MIN_WORD_SEC = CTC «утрамбовал» текст без опоры в аудио (чтец пропустил кусок
    или диапазон аятов взят с лишком). Работает прямо по CTC-границам (до сборки word_timeline)
    и возвращает ГРАНИЦЫ прогонов, а не плоский список — так их можно не ронять, а растянуть
    по дыре (`_respace_stuffed`). Одиночные короткие слова не трогаем."""
    runs: list[tuple[int, int]] = []
    run: list[int] = []
    for i, (a, b) in enumerate(bounds):
        if (b - a) < MIN_WORD_SEC:
            run.append(i)
        else:
            if len(run) >= STUFFED_RUN:
                runs.append((run[0], run[-1]))
            run = []
    if len(run) >= STUFFED_RUN:
        runs.append((run[0], run[-1]))
    return runs


def _respace_stuffed(bounds: list[tuple[float, float]], ref: list[tuple],
                     runs: list[tuple[int, int]]) -> tuple[list[tuple[float, float]], int]:
    """Растянуть тайминги скомканных слов по дыре ВМЕСТО молчаливого отсева.

    Симптом (rec11, Ан-Наджм 53:3→4): чтец тянет الْهَوَىٰ ~4с (мadd), CTC под-аллоцирует
    ему ~0.24с, а слова 53:4 (إِنْ هُوَ إِلَّا وَحْيٌ) утрамбовывает в десятки мс и старое
    правило плотности их РОНЯЛО → в word_timeline 6.4с-дыра → подсветка залипает на الهوى.
    Здесь мы вместо отсева раскладываем скомканный прогон по свободному интервалу, чтобы
    подсветка ехала по словам (общий класс «forced скомкал кусок»).

    Раскладка: старт = где CTC поставил прогон, но не раньше конца пред. слова (onset);
    так «долгое» пред. слово держит фокус до момента, куда CTC поместил утрамбованный кусок,
    и только потом подсветка идёт по нему. Если от onset до следующей опоры места мало
    (< N×MIN_WORD_SEC) — берём всю дыру от конца пред. слова. Время делим пропорционально
    длине согласного скелета слова (длинные держатся дольше). Монотонность сохраняется.

    Возвращает (новые bounds, число растянутых слов)."""
    bounds = list(bounds)
    n = len(bounds)
    total = 0
    for i0, i1 in runs:
        L = bounds[i0 - 1][1] if i0 > 0 else bounds[i0][0]
        R = bounds[i1 + 1][0] if i1 + 1 < n else bounds[i1][1]
        cnt = i1 - i0 + 1
        onset = max(L, bounds[i0][0])
        if R - onset < cnt * MIN_WORD_SEC:
            onset = L                      # места от CTC-позиции мало → вся дыра
        span = R - onset
        if span <= 0:
            continue                       # нет места (край/инверсия) — оставляем как есть
        weights = []
        for j in range(i0, i1 + 1):
            w = ref[j][3] if j < len(ref) else ""
            weights.append(sum(1 for ch in w if ch not in _HARAKAT) or 1)
        wsum = sum(weights)
        cur = onset
        for k, j in enumerate(range(i0, i1 + 1)):
            a = cur
            b = R if j == i1 else cur + span * weights[k] / wsum
            bounds[j] = (a, b)
            cur = b
            total += 1
    return bounds, total


# --- подтяжка границ слов к тишине (наследие quran-align, boundaries.cc) ------------------
# CTC даёт границы слова с точностью до кадра эмиссий (~40 мс) и нередко «прихватывает» тишину
# по краям: старт слова заезжает в предшествующую паузу (→ подсветка/скролл прыгает на слово
# ДО того, как чтец его начал — семья бага ложного раннего якоря 25:65→66), конец висит в
# последующей тишине. Пост-шаг поджимает границы ВНУТРЬ к реальной речи по RMS-огибающей.
# Порог тишины калибруем от шумового пола КОНКРЕТНОЙ записи (перцентиль) — у нас YouTube-читки
# с разным фоном, фикс-пороги quran-align (−100/−75 dBFS) заточены под студийный мураттал.
# Двигаем ТОЛЬКО внутрь (старт не раньше, конец не позже исходного CTC) → нельзя заехать на
# речь соседнего слова, монотонность таймлайна сохраняется. Полностью тихое слово не трогаем.
SAMPLE_RATE = 16000          # cfa.load_audio всегда ресемплит в 16 кГц моно
_SNAP_FRAME_MS = 20          # кадр RMS-огибающей
_SNAP_WINDOW_SEC = 0.30      # насколько далеко ищем речь/паузу от исходной границы
_SNAP_MARGIN_DB = 10.0       # порог речи = шумовой пол + запас
_SNAP_FLOOR_PCT = 15         # перцентиль кадров для оценки шумового пола
_SNAP_MIN_RUN = 3            # столько подряд кадров речи/тишины подтверждают переход (гистерезис)
_SNAP_MIN_SHIFT_SEC = 0.03   # меньше — не считаем подтяжкой (шум округления)
_SNAP_MIN_WORD_SEC = 0.04    # не сжимать слово короче этого


def _frame_db(wav, frame_len: int):
    """Поканальный RMS в dBFS по не перекрывающимся кадрам."""
    import numpy as np
    n = len(wav) // frame_len
    if n == 0:
        return None
    frames = np.asarray(wav[: n * frame_len], dtype=np.float32).reshape(n, frame_len)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-14)
    return 20.0 * np.log10(np.maximum(rms, 1e-7))


def _snap_bounds(bounds: list[tuple[float, float]], wav):
    """Поджать [(t0,t1)] к речи по RMS-огибающей. Возвращает (new_bounds, n_snapped).

    Fail-safe: при любой проблеме (нет numpy / пустое аудио / вырожденный порог) возвращает
    исходные границы без изменений — пост-шаг не должен ронять forced align."""
    try:
        import numpy as np
        frame_len = max(1, int(SAMPLE_RATE * _SNAP_FRAME_MS / 1000))
        db = _frame_db(wav, frame_len)
        if db is None or len(db) < _SNAP_MIN_RUN:
            return bounds, 0
        floor = float(np.percentile(db, _SNAP_FLOOR_PCT))
        thr = floor + _SNAP_MARGIN_DB
        speech = db >= thr
        if not speech.any() or speech.all():
            return bounds, 0        # вся запись «речь» или «тишина» → порог бесполезен
        frame_sec = frame_len / SAMPLE_RATE
        win = max(1, int(_SNAP_WINDOW_SEC / frame_sec))
        nf = len(speech)

        def confirmed_speech(i):
            """Речь, подтверждённая _SNAP_MIN_RUN кадрами вперёд от i."""
            return 0 <= i < nf and speech[i] and speech[i:i + _SNAP_MIN_RUN].all()

        def confirmed_speech_back(i):
            """Речь, подтверждённая _SNAP_MIN_RUN кадрами назад от i (включительно)."""
            return 0 <= i < nf and speech[i] and speech[max(0, i - _SNAP_MIN_RUN + 1):i + 1].all()

        out, n_snapped = [], 0
        for t0, t1 in bounds:
            a = int(round(t0 / frame_sec))
            b = int(round(t1 / frame_sec))
            nt0, nt1 = t0, t1
            # СТАРТ: если начало в тишине — сдвинуть вперёд к первому подтверждённому кадру речи
            if not confirmed_speech(min(a, nf - 1)):
                for j in range(max(0, a), min(nf, a + win + 1)):
                    if confirmed_speech(j):
                        cand = j * frame_sec
                        if cand > t0 and cand < t1 - _SNAP_MIN_WORD_SEC:
                            nt0 = cand
                        break
            # КОНЕЦ: если конец в тишине — подтянуть назад к последнему подтверждённому кадру речи
            if not confirmed_speech_back(min(b - 1, nf - 1)):
                for j in range(min(nf - 1, b - 1), max(-1, b - win - 1), -1):
                    if confirmed_speech_back(j):
                        cand = (j + 1) * frame_sec
                        if cand < t1 and cand > nt0 + _SNAP_MIN_WORD_SEC:
                            nt1 = cand
                        break
            if (nt0 - t0) >= _SNAP_MIN_SHIFT_SEC or (t1 - nt1) >= _SNAP_MIN_SHIFT_SEC:
                n_snapped += 1
            out.append((nt0, nt1))
        return out, n_snapped
    except Exception:
        return bounds, 0


# --- детект возвратов чтеца (П8) -----------------------------------------------------------
# Forced-align МОНОТОНЕН: проходит текст слева направо один раз и физически не выражает повтор.
# Но чтецы в таджвиде постоянно останавливаются, отходят на пару слов назад и ПЕРЕЧИТЫВАЮТ.
# В монотонном таймлайне это выглядит как «дыра»: между словом wi и wi+1 висит несколько секунд
# аудио, которому не досталось ни одного эталонного слова (повтор некуда было отобразить) →
# подсветка/скролл замирают, а вживую чтец уже читает заново. Здесь — отдельный ПОСТ-шаг:
# по акустике находим такие дыры-повторы и вклеиваем в word_timeline «назадние» точки (их wi
# меньше текущего). Фронт ищет активное слово бинпоиском по t и берёт wi — при монотонном t
# корпус-позиция может идти назад, поэтому подсветка честно возвращается, а в конце дыры
# перескакивает вперёд на wi+1 (где чтец возобновил). Фронт менять не надо.
#
# Как ловим (акустика + пауза, БЕЗ google/whisper — текст ASR повтор часто не пишет разборчиво):
#   1. дыра = разрыв ≥ _REPEAT_MIN_GAP между концом слова и началом следующего;
#   2. в разрыве должна быть РЕЧЬ (доля кадров выше шумового порога ≥ _REPEAT_MIN_SPEECH) —
#      иначе это обычная пауза/вдох, не повтор;
#   3. greedy-CTC-декод эмиссий разрыва → романизованные буквы MMS → согласный СКЕЛЕТ (greedy
#      роняет гласные, поэтому и эталон сводим к скелету, как в align.py); тандемный повтор в декоде
#      (чтец без паузы прочёл фразу дважды → скелет `PP`) схлопываем в `P` (_collapse_tandem), иначе
#      удвоенная длина тянет матч на лишний аят назад (over-reach);
#   4. ищем «назадний» непрерывный диапазон эталона [ra..rb] (rb ≤ текущего слова, ra не дальше
#      _REPEAT_LOOKBACK слов назад), чей скелет ближе всего к декоду разрыва; берём, если похоже
#      (≥ _REPEAT_MIN_SIM) и явно лучше, чем «вперёд» (чтобы не спутать возврат с медленным
#      началом следующего слова). Время разрыва раздаём словам диапазона по длине их скелета.
# Опт-аут: SYNC_FALIGN_REPEATS=0. Fail-safe: любая ошибка → просто нет вставок (align не падает).
_REPEAT_MIN_GAP = 0.7        # сек: короче — не считаем разрывом (внутрисловные паузы таджвида)
_REPEAT_MIN_SPEECH = 0.35    # доля кадров-речи в разрыве (иначе пауза/вдох, не повтор)
_REPEAT_MIN_DECODE = 4       # символов скелета в декоде разрыва (меньше — шум, не слово)
_REPEAT_LOOKBACK = 12        # на сколько эталонных слов назад ищем повтор (хватает на 1-2 аята)
_REPEAT_MIN_SIM = 0.5        # порог похожести скелетов (Левенштейн-similarity)
_REPEAT_FWD_MARGIN = 0.05    # «назад» должно быть лучше «вперёд» хотя бы на столько
_REPEAT_PAUSE_MIN = 0.15     # мин. длительность паузы в разрыве, чтобы якорить повтор на её конец
_VOWELS = set("aeiou")


def _skeleton(s: str) -> str:
    """Согласный скелет романизованной строки (гласные и апостроф долой)."""
    return "".join(c for c in s if c not in _VOWELS and c != "'")


def _uroman_word(w: str) -> str:
    """Романизация одного арабского слова так же, как ctc-forced-aligner (unidecode+uroman)."""
    import ctc_forced_aligner as cfa
    from unidecode import unidecode
    return cfa.normalize_uroman(unidecode(w)).replace(" ", "")


def _lev(a: str, b: str) -> int:
    """Расстояние Левенштейна (итеративное, O(len(a)*len(b)) памяти O(len(b)))."""
    m, n = len(a), len(b)
    if not m or not n:
        return max(m, n)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[n]


def _sim(a: str, b: str) -> float:
    return 1.0 - _lev(a, b) / max(len(a), len(b), 1)


def _collapse_tandem(s: str) -> str:
    """Схлопнуть тандемный повтор в начале декода разрыва (чтец дважды прочёл одну фразу подряд —
    без паузы между копиями, поэтому onset их не разделил → greedy-CTC выдал удвоенный скелет `PP…`).
    Удвоенная длина тянет назадний матч на ЛИШНИЙ аят (длиннее эталон-диапазон = ближе по длине к
    раздутому декоду), т.е. ложный over-reach (rec11 ~0:50: чтец перечёл 53:11 дважды, декод удвоился,
    матч уехал на 53:10 вместо 53:11). Ищем период p, при котором ≥2 идущих подряд блока длины p
    почти равны первому (Левенштейн ≥0.6 — CTC шумит), оставляем ОДИН блок + хвост. Только укорачивает
    и только при явном тандеме → чистые (неудвоенные) декоды не трогает."""
    n = len(s)
    if n < 4:
        return s
    best = s
    for p in range(2, n // 2 + 1):
        b0 = s[:p]
        reps = 1
        while (reps + 1) * p <= n and _sim(b0, s[reps * p:(reps + 1) * p]) >= 0.6:
            reps += 1
        if reps >= 2:
            cand = b0 + s[reps * p:]
            if len(cand) < len(best):
                best = cand
    return best


def _greedy_ctc(emissions, stride_ms: int, t0: float, t1: float, id2ch: dict) -> str:
    """Greedy-CTC-декод эмиссий во временном окне [t0,t1) → романизованные буквы (collapse+blank).
    Последний столбец эмиссий — служебный <star> (добавлен нулями, logprob 0 → всегда выигрывал
    бы argmax над реальными лог-вероятностями ≤0), поэтому argmax берём по [:, :31]."""
    f0 = max(0, int(t0 * 1000 / stride_ms))
    f1 = min(emissions.shape[0], int(t1 * 1000 / stride_ms))
    if f1 <= f0:
        return ""
    ids = emissions[f0:f1, :31].argmax(axis=1)
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i > 3:      # 0..3 = blank/pad/eos/unk; 4..30 = буквы
            ch = id2ch.get(i, "")
            if len(ch) == 1:
                out.append(ch)
        prev = i
    return "".join(out)


def _detect_repeats(bounds, ref, wav, emissions, stride_ms):
    """Найти возвраты чтеца → список word_timeline-точек для вклейки (их wi идёт «назад»).

    bounds[i]=(t0,t1) — границы эталонного слова ref[i]=(surah,ayah,wi,arabic), n монотонных слов.
    Возвращает (inserts, details): inserts — точки word_timeline (wi «назад»); details — по одному
    диагно-словарю на КАЖДЫЙ сработавший возврат (onset/gap/decode/sim/from/back) для аудита без звука
    и без повторного прогона GPU (кладётся в meta.repeats_detail). При любой проблеме → ([], [])."""
    try:
        import numpy as np
        import ctc_forced_aligner as cfa
        n = len(bounds)
        if n < 2 or emissions is None:
            return [], []
        id2ch = {v: k for k, v in cfa.VOCAB_DICT.items()}
        skel = [_skeleton(_uroman_word(ref[i][3])) for i in range(n)]

        # маска речи по RMS-огибающей (тот же порог, что в _snap_bounds)
        frame_len = max(1, int(SAMPLE_RATE * _SNAP_FRAME_MS / 1000))
        db = _frame_db(wav, frame_len)
        speech = None
        if db is not None and len(db) >= _SNAP_MIN_RUN:
            thr = float(np.percentile(db, _SNAP_FLOOR_PCT)) + _SNAP_MARGIN_DB
            speech = db >= thr
        frame_sec = frame_len / SAMPLE_RATE

        def speech_frac(t0, t1):
            if speech is None:
                return 1.0
            a, b = int(t0 / frame_sec), int(t1 / frame_sec)
            a, b = max(0, a), min(len(speech), b)
            return float(speech[a:b].mean()) if b > a else 0.0

        def repeat_onset(t0, t1):
            """Начало повторного чтения в разрыве = конец САМОЙ ДЛИННОЙ паузы (тишины) внутри [t0,t1].
            До паузы держим слово-остановку (чтец договаривает/молчит), после — возврат. Так خَضِرًا
            не «обделён фокусом»: подсветка стоит на нём сквозь паузу, а назад прыгает только когда
            реально зазвучал повтор. Явной паузы нет (< _REPEAT_PAUSE_MIN) → фолбэк на t0."""
            if speech is None:
                return t0
            a, b = max(0, int(t0 / frame_sec)), min(len(speech), int(t1 / frame_sec))
            if b - a < 2:
                return t0
            seg = speech[a:b]
            best_len, best_end, run = 0, a, 0
            for j in range(len(seg)):
                if not seg[j]:
                    run += 1
                else:
                    if run > best_len:          # серия тишины кончилась речью → кандидат-онсет
                        best_len, best_end = run, a + j
                    run = 0
            if best_len * frame_sec < _REPEAT_PAUSE_MIN:
                return t0
            return best_end * frame_sec

        inserts, details = [], []
        for i in range(n - 1):
            g0, g1 = bounds[i][1], bounds[i + 1][0]
            if g1 - g0 < _REPEAT_MIN_GAP:
                continue
            # якорь = начало повторного чтения (после паузы). Интервал [g0, onset] остаётся
            # слову-остановке (держим подсветку, не режем его), декод/раздача — от onset.
            onset = repeat_onset(g0, g1)
            # речь проверяем в ПОВТОРНОЙ части [onset, g1] (после паузы), а НЕ во всей дыре: дыра
            # часто = длинная пауза + короткая перечитка → доля речи по всей дыре низкая (rec11 01:24:
            # 0.27), а после паузы высокая. Точнее и безопаснее глобального понижения порога.
            if speech_frac(onset, g1) < _REPEAT_MIN_SPEECH:
                continue                              # и после паузы тихо — обычная пауза, не повтор
            dec = _collapse_tandem(_skeleton(_greedy_ctc(emissions, stride_ms, onset, g1, id2ch)))
            if len(dec) < _REPEAT_MIN_DECODE:
                continue
            # лучший «назадний» непрерывный диапазон эталона [ra..rb], rb ≤ i.
            # КРОСС-АЯТНЫЕ возвраты разрешены (чтец может отойти на аят-два назад и перечитать —
            # rec11 ~0:50: вернулся к 53:10 и перечитал до 53:12, декод дыры дублирует скелет 53:11).
            # Раньше диапазон ограничивался ОДНИМ аятом (защита от «свипов» плывущего базового
            # выравнивания) — но это подавляло реальные кросс-аятные возвраты (владелец подтвердил ухом).
            # От дрейфа защищает forward-margin-гард ниже: чистый дрейф = forward-контент → отсекается,
            # реальный возврат = дублированный декод, «вперёд» не совпадает → проходит. Опт-аут на
            # старое поведение (в пределах аята): SYNC_FALIGN_REPEAT_XAYAH=0.
            xayah = os.environ.get("SYNC_FALIGN_REPEAT_XAYAH", "1") != "0"
            best = None
            for ra in range(max(0, i - _REPEAT_LOOKBACK), i + 1):
                cand = ""
                for rb in range(ra, i + 1):
                    if not xayah and (ref[rb][0] != ref[ra][0] or ref[rb][1] != ref[ra][1]):
                        break                     # (опт-аут) вышли за пределы аята слова ra
                    cand += skel[rb]
                    s = _sim(dec, cand)
                    if best is None or s > best[0]:
                        best = (s, ra, rb)
            if best is None or best[0] < _REPEAT_MIN_SIM:
                continue
            # защита от ложняка (и от дрейфа базы): «вперёд» (следующие слова) не должно совпадать
            # лучше «назад» — иначе это не возврат, а медленное/сдвинутое чтение вперёд
            fwd = ""
            for k in range(i + 1, min(n, i + 1 + (best[2] - best[1] + 1))):
                fwd += skel[k]
            if fwd and _sim(dec, fwd) >= best[0] - _REPEAT_FWD_MARGIN:
                continue
            sim, ra, rb = best
            # диагностика возврата — для tools/audit_repeats.py (объективно, без звука/GPU-прогона)
            details.append({
                "onset": round(onset, 2), "gap": round(g1 - onset, 2),
                "decode": dec, "sim": round(sim, 3),
                "from": f"{ref[i][0]}:{ref[i][1]}:{ref[i][2]}",
                "back": f"{ref[ra][0]}:{ref[ra][1]}:{ref[ra][2]}"
                        f"..{ref[rb][0]}:{ref[rb][1]}:{ref[rb][2]}"})
            # раздать время ПОВТОРА (от onset, не от g0) словам [ra..rb] пропорц. длине их скелета
            lens = [max(1, len(skel[k])) for k in range(ra, rb + 1)]
            total = sum(lens)
            span = g1 - onset
            acc = onset
            for off, k in enumerate(range(ra, rb + 1)):
                wt0 = acc
                acc += span * lens[off] / total
                su, ay, wi, _ = ref[k]
                inserts.append({"t": round(wt0, 3), "t_end": round(acc, 3),
                                "surah": su, "ayah": ay, "wi": wi, "rep": True})
        return inserts, details
    except Exception:
        return [], []


def available() -> bool:
    """Доступны ли зависимости forced align в этом окружении (CPU-докер их не ставит).
    Лёгкая проверка импортируемости — без загрузки модели (её тянет только align())."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("ctc_forced_aligner", "onnxruntime"))


@lru_cache(maxsize=1)
def _session_and_tokenizer():
    import ctc_forced_aligner as cfa
    import onnxruntime
    model_path = os.path.expanduser("~/ctc_forced_aligner/model.onnx")
    cfa.ensure_onnx_model(model_path, cfa.MODEL_URL)
    avail = onnxruntime.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail \
        else ["CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(model_path, providers=providers)
    return session, cfa.Tokenizer(), cfa


def align(audio_path, verses: list[tuple[int, int, str]], batch_size: int | None = None) -> dict:
    """Forced alignment диапазона аятов к аудио.

    verses — [(surah, ayah, text), ...] по порядку чтения (текст из quran.db, с харакат).
    Возвращает sync_map: {meta, timeline, word_timeline, char_timeline}.

    batch_size — сколько 30-с окон гнать через wav2vec2 за один session.run. На GPU память
    ~线ейна по batch_size×окно (feature-extractor layer-norm аллоцирует буфер на весь батч):
    RTX 3060 (6 ГБ): batch_size=8 (~272с разом) падает OOM (7 ГБ); даже batch_size=2 (1.78 ГБ)
    падал при свободных 5.7 ГБ (арена onnxruntime растёт неудачно). batch_size=1 стабилен и
    быстр — forced rec10 (654с аудио) = 17.6с на GPU (против ~6 мин на CPU). Дефолт 1.
    Переопределяется env SYNC_FALIGN_BATCH. На CPU значение почти не важно (память хостовая).
    """
    if batch_size is None:
        batch_size = int(os.environ.get("SYNC_FALIGN_BATCH", "1"))
    session, tokenizer, cfa = _session_and_tokenizer()

    # эталон: плоский список слов с привязкой к аяту; индекс слова = word_index в аяте
    ref = []  # (surah, ayah, wi, arabic_word)
    for surah, ayah, txt in verses:
        for wi, w in enumerate(txt.split()):
            ref.append((surah, ayah, wi, w))
    text = " ".join(txt for _, _, txt in verses)

    wav = cfa.load_audio(str(audio_path))
    emissions, stride = cfa.generate_emissions(session, wav, batch_size=batch_size)
    tokens_starred, text_starred = cfa.preprocess_text(text, romanize=True, language="ara")
    segments, scores, blank = cfa.get_alignments(emissions, tokens_starred, tokenizer)
    spans = cfa.get_spans(tokens_starred, segments, blank)
    wts = cfa.postprocess_results(text_starred, spans, stride, scores)

    # выкинуть служебные звёздочки-разделители → должно остаться 1:1 с ref
    units = [w for w in wts if w.get("text") not in ("<star>", "*", "")]
    n = min(len(units), len(ref))

    # подтяжка границ слов к тишине по RMS-огибающей (опт-аут SYNC_FALIGN_SNAP=0)
    bounds = [(float(units[i]["start"]), float(units[i]["end"])) for i in range(n)]
    snapped = 0
    if os.environ.get("SYNC_FALIGN_SNAP", "1") != "0":
        bounds, snapped = _snap_bounds(bounds, wav)

    # правило плотности (quran-align): «скомканные» участки = текст без опоры в аудио.
    # По умолчанию (task2 идея 1) НЕ роняем их молча — иначе в word_timeline остаётся дыра и
    # подсветка залипает на пред. слове; вместо этого растягиваем тайминги прогона по дыре
    # (подсветка едет по словам). Опт-аут SYNC_FALIGN_RESPACE=0 → старое поведение (отсев).
    # Сырые границы до растяжки сохраняем для детекта повторов (П8) — чтобы не задеть его валидацию.
    raw_bounds = list(bounds)
    runs = _stuffed_runs(bounds)
    respaced = 0
    stuffed_idx: list[int] = []
    if runs and os.environ.get("SYNC_FALIGN_RESPACE", "1") != "0":
        bounds, respaced = _respace_stuffed(bounds, ref, runs)
    elif runs:
        stuffed_idx = [i for i0, i1 in runs for i in range(i0, i1 + 1)]

    word_timeline, timeline, char_timeline = [], [], []
    seen_ayah = set()
    for i in range(n):
        surah, ayah, wi, arabic = ref[i]
        t0, t1 = bounds[i]
        word_timeline.append({"t": round(t0, 3), "t_end": round(t1, 3),
                              "surah": surah, "ayah": ayah, "wi": wi})
        if (surah, ayah) not in seen_ayah:
            seen_ayah.add((surah, ayah))
            timeline.append({"t": round(t0, 3), "surah": surah, "ayah": ayah})
        # посимвольная дорожка: интерполяция внутри точных границ слова.
        # харакат наследуют время своей базовой буквы (нулевая доля длительности).
        base_positions = [j for j, ch in enumerate(arabic) if ch not in _HARAKAT]
        nb = max(1, len(base_positions))
        for ci, ch in enumerate(arabic):
            # доля позиции среди базовых букв (combining-марка = как её база)
            k = sum(1 for p in base_positions if p < ci)
            frac0 = k / nb
            frac1 = (k + 1) / nb
            ct0 = t0 + (t1 - t0) * frac0
            ct1 = t0 + (t1 - t0) * (frac0 if ch in _HARAKAT else frac1)
            char_timeline.append({"t": round(ct0, 3), "t_end": round(ct1, 3),
                                   "surah": surah, "ayah": ayah, "wi": wi, "ci": ci})

    # старый путь (SYNC_FALIGN_RESPACE=0): выкинуть «скомканные» участки вместо растяжки
    if stuffed_idx:
        drop = {(word_timeline[i]["surah"], word_timeline[i]["ayah"], word_timeline[i]["wi"])
                for i in stuffed_idx}
        word_timeline = [w for w in word_timeline if (w["surah"], w["ayah"], w["wi"]) not in drop]
        char_timeline = [c for c in char_timeline if (c["surah"], c["ayah"], c["wi"]) not in drop]
        timeline, seen_ayah = [], set()   # старт аята мог выпасть — пересобрать по выжившим
        for w in word_timeline:
            k = (w["surah"], w["ayah"])
            if k not in seen_ayah:
                seen_ayah.add(k)
                timeline.append({"t": w["t"], "surah": w["surah"], "ayah": w["ayah"]})

    # детект возвратов чтеца (П8): вклеить «назадние» точки в дыры-повторы (опт-аут SYNC_FALIGN_REPEATS=0).
    # Кормим СЫРЫМИ границами (до растяжки скомканных) — стянутые дыры не должны влиять на П8.
    repeats, repeats_detail = [], []
    if os.environ.get("SYNC_FALIGN_REPEATS", "1") != "0":
        repeats, repeats_detail = _detect_repeats(raw_bounds, ref, wav, emissions, stride)
        if repeats:
            word_timeline.extend(repeats)
            word_timeline.sort(key=lambda w: (w["t"], w["surah"], w["ayah"], w["wi"]))

    meta = {
        "aligner": "forced-mms-ctc",
        "ref_words": len(ref),
        "aligned_units": len(units),
        "coverage": round(n / len(ref), 3) if ref else 0.0,
        "stuffed_dropped": len(stuffed_idx),
        "stuffed_respaced": respaced,
        "snapped_to_silence": snapped,
        "repeats_inserted": len(repeats),
        "repeats_detail": repeats_detail,
        "wt": len(word_timeline),
        "ct": len(char_timeline),
        "providers": session.get_providers(),
    }
    return {"meta": meta, "timeline": timeline,
            "word_timeline": word_timeline, "char_timeline": char_timeline}


def verses_from_data(data: dict, min_edge_coverage: float = 0.5) -> list[tuple[int, int, str]]:
    """Достать диапазон (surah, ayah, text) из data.json уже готового прогона (google/whisper),
    где align.py определил, какие аяты читаются.

    Обрезка КРАЕВЫХ аятов без реальной опоры (баг 3b): если исходный ASR заякорил у ведущего/
    хвостового аята малую долю слов (< min_edge_coverage) — значит аудио содержит его лишь
    частично/вовсе нет, и forced скомкает полный текст в пару секунд (мусор в начале). Считаем
    покрытие по числу РАЗНЫХ wi в word_timeline источника (интерполяция не выходит за первый/
    последний якорь, поэтому на краях покрытие честное). Режем только с краёв, до первого
    хорошо покрытого аята; внутренние не трогаем. Порог с большим запасом (rec5: фантом 0.15 при
    ближайшем легитимном 0.83), rec6/rec7 не задеты.
    """
    verses = []  # (surah, ayah, text, nwords)
    for sec in data.get("sections", []):
        s = sec["surah"]
        for v in sec.get("ayat", []):
            nwords = len(v.get("words") or v["text"].split())
            verses.append((s, v["ayah"], v["text"], nwords))
    if not verses:
        return []

    wt = data.get("word_timeline") or []
    if wt and min_edge_coverage > 0:
        from collections import defaultdict
        anchored: dict[tuple[int, int], set] = defaultdict(set)
        for w in wt:
            anchored[(w["surah"], w["ayah"])].add(w["wi"])

        def covered(v) -> float:
            su, ay, _, n = v
            return len(anchored.get((su, ay), ())) / n if n else 1.0

        lo, hi = 0, len(verses)
        while lo < hi - 1 and covered(verses[lo]) < min_edge_coverage:
            lo += 1
        while hi - 1 > lo and covered(verses[hi - 1]) < min_edge_coverage:
            hi -= 1
        verses = verses[lo:hi]

    return [(su, ay, txt) for su, ay, txt, _ in verses]
