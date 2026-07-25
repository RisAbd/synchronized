"""Общие кирпичи выравнивания — самодостаточные DSP/строковые хелперы БЕЗ привязки к источнику.

Директива владельца (24-25.07): каждый распознаватель/выравниватель независим, ноль наследования
данных между источниками. Но чисто ВЫЧИСЛИТЕЛЬНЫЕ кирпичи (RMS-огибающая, подтяжка границ к тишине,
Левенштейн, схлопывание тандема) — это не данные источника, а общая математика. Их держим здесь и
повторяем в каждом источнике импортом, а не копипастой (владелец: «всю эту логику вытащи в один
модуль»). Так `w2v_*` больше не импортирует `falign` ради математики — связь источников разорвана.

Ничего тяжёлого на уровне модуля (numpy импортируется лениво внутри функций) — можно тянуть из
любого источника без побочных загрузок моделей.
"""
from __future__ import annotations

# --- арабские строки -----------------------------------------------------------------------
# combining-марки (харакат/шадда/сукун/танвин/мадда и т.п.) — «висят» на предыдущей букве
_HARAKAT = set("ًٌٍَُِّْٰٕٖٓٔٗ٘")


def _lev(a: str, b: str) -> int:
    """Расстояние Левенштейна (итеративное, O(len(a)*len(b)) времени, O(len(b)) памяти)."""
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


# --- подтяжка границ слов к тишине (наследие quran-align, boundaries.cc) --------------------
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
