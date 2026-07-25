"""Источник w2v — wav2vec2 + СВОЙ CTC-Viterbi (без whisperx). ПОЛНОСТЬЮ независим (директива
владельца 24.07): audio → своя акустика (эмиссии) → сам находит диапазон аятов (`match_align.
find_range`) → свой монотонный force-align этого диапазона по тем же эмиссиям → свои возвраты
чтеца (`w2v_repeats`). ASR ему НЕ нужен, данные других источников НЕ наследует. Держит слово сквозь
мадд → честный coverage; дефолтный выравниватель в плеере (PRIORITY выше forced).

ISOLATE=True: гоняется в отдельном процессе (gpu_align) — torch(w2v) и onnxruntime(forced) в одном
процессе на 6ГБ = OOM (липкая CUDA-арена). AUTO=True: авто-пост-шаг на каждой записи."""
from __future__ import annotations

import json
from pathlib import Path

KEY = "w2v"
LABEL = "Forced align (wav2vec2)"
NOTE = ("wav2vec2 + СВОЙ CTC-Viterbi (без whisperx): держит слово сквозь мадд → честный coverage; "
        "ПОЛНОСТЬЮ независим — сам находит диапазон из акустики (ASR не нужен); активен по умолчанию")
SELECTABLE = False
AUTO = True
ISOLATE = True
ALIGNED = True
PRIORITY = 10            # выше forced → дефолт в плеере среди выравнивателей


def available() -> bool:
    try:
        import w2v_align
        return w2v_align.available()
    except Exception:
        return False


def run(rec, audio, quran, out_dir: Path, stage=None) -> dict:
    """Один GPU-проход: эмиссии → диапазон → force-align → возвраты. Зовётся из подпроцесса
    gpu_align (ISOLATE). `quran` — экземпляр Quran; `audio` — Path к аудио записи."""
    import match_align
    import w2v_align
    import w2v_repeats

    E, stride, idx2ch, ch2idx = w2v_align.emissions(str(audio))
    index = match_align.build_index(quran)
    rng = match_align.find_range(E, quran, idx2ch, ch2idx, index=index)   # [(surah, ayah), ...]
    if not rng:
        raise RuntimeError("w2v: не удалось определить диапазон из акустики")
    verses = [(s, a, quran.surah(s).verses[a - 1].text) for s, a in rng]
    # свой монотонный CTC-Viterbi по УЖЕ посчитанным эмиссиям (окна не нужны — путь глобален)
    sync_map = w2v_align.forced_align(E, stride, verses, idx2ch, ch2idx, str(audio))
    # возвраты/перечитки чтеца (П8) из СВОЕЙ акустики w2v (не наследуем у forced) — вклейка rep-точек
    reps = w2v_repeats.detect(E, stride, idx2ch, ch2idx,
                              sync_map["word_timeline"], verses, str(audio))
    if reps:
        sync_map["word_timeline"].extend(reps)
        sync_map["word_timeline"].sort(key=lambda w: (w["t"], w["surah"], w["ayah"], w["wi"]))
    meta = sync_map.setdefault("meta", {})
    meta["range_source"] = "w2v-self"
    meta["range"] = f"{rng[0][0]}:{rng[0][1]}..{rng[-1][0]}:{rng[-1][1]}"
    meta["repeats_inserted"] = len(reps)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return sync_map
