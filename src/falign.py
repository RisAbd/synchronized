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


def _stuffed_indices(word_timeline: list[dict]) -> list[int]:
    """Индексы слов в «скомканных» участках: ≥ STUFFED_RUN подряд слов короче MIN_WORD_SEC.

    Симптом: чтец пропустил кусок (или диапазон аятов определился с лишком, который не
    поймала краевая обрезка verses_from_data), а CTC обязан разместить ВЕСЬ текст → лишние
    слова сжимаются в десятки миллисекунд. В quran-align это же лечится отсевом спанов без
    опоры распознавания короче k×100 мс. Одиночные короткие слова не трогаем."""
    bad, run = [], []
    for i, w in enumerate(word_timeline):
        # слова без t_end (таймлайны align.py интерполированы) длительности не имеют — пропуск
        if w.get("t_end") is not None and (w["t_end"] - w["t"]) < MIN_WORD_SEC:
            run.append(i)
        else:
            if len(run) >= STUFFED_RUN:
                bad.extend(run)
            run = []
    if len(run) >= STUFFED_RUN:
        bad.extend(run)
    return bad


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

    # правило плотности (quran-align): выкинуть «скомканные» участки — текст без опоры в аудио
    stuffed = _stuffed_indices(word_timeline)
    if stuffed:
        drop = {(word_timeline[i]["surah"], word_timeline[i]["ayah"], word_timeline[i]["wi"])
                for i in stuffed}
        word_timeline = [w for w in word_timeline if (w["surah"], w["ayah"], w["wi"]) not in drop]
        char_timeline = [c for c in char_timeline if (c["surah"], c["ayah"], c["wi"]) not in drop]
        timeline, seen_ayah = [], set()   # старт аята мог выпасть — пересобрать по выжившим
        for w in word_timeline:
            k = (w["surah"], w["ayah"])
            if k not in seen_ayah:
                seen_ayah.add(k)
                timeline.append({"t": w["t"], "surah": w["surah"], "ayah": w["ayah"]})

    meta = {
        "aligner": "forced-mms-ctc",
        "ref_words": len(ref),
        "aligned_units": len(units),
        "coverage": round(n / len(ref), 3) if ref else 0.0,
        "stuffed_dropped": len(stuffed),
        "snapped_to_silence": snapped,
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
