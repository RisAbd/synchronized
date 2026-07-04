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

    meta = {
        "aligner": "forced-mms-ctc",
        "ref_words": len(ref),
        "aligned_units": len(units),
        "coverage": round(n / len(ref), 3) if ref else 0.0,
        "wt": len(word_timeline),
        "ct": len(char_timeline),
        "providers": session.get_providers(),
    }
    return {"meta": meta, "timeline": timeline,
            "word_timeline": word_timeline, "char_timeline": char_timeline}


def verses_from_data(data: dict) -> list[tuple[int, int, str]]:
    """Достать диапазон (surah, ayah, text) из data.json уже готового прогона (google/whisper),
    где align.py определил, какие аяты читаются."""
    out = []
    for sec in data.get("sections", []):
        s = sec["surah"]
        for v in sec.get("ayat", []):
            out.append((s, v["ayah"], v["text"]))
    return out
