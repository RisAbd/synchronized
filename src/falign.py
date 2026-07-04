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


def align(audio_path, verses: list[tuple[int, int, str]], batch_size: int = 8) -> dict:
    """Forced alignment диапазона аятов к аудио.

    verses — [(surah, ayah, text), ...] по порядку чтения (текст из quran.db, с харакат).
    Возвращает sync_map: {meta, timeline, word_timeline, char_timeline}.
    """
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

    word_timeline, timeline, char_timeline = [], [], []
    seen_ayah = set()
    for i in range(n):
        surah, ayah, wi, arabic = ref[i]
        u = units[i]
        t0, t1 = float(u["start"]), float(u["end"])
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
