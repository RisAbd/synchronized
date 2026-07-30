"""Источник forced — MMS CTC forced alignment. ПОЛНОСТЬЮ независим (директива владельца 24.07:
каждый источник независим, ноль наследования между источниками): audio → свои MMS-эмиссии →
САМ находит диапазон аятов (романизованный greedy-декод → `match_align.find_range` по
романизованному индексу корпуса) → force-align этого диапазона по тем же эмиссиям. ASR ему больше
НЕ нужен (снят `pipeline._forced_source` + гейт `ready()` — инкремент 3). Возвраты чтеца (П8)
детектит свой `falign._detect_repeats` по MMS-эмиссиям внутри align_verses.

ISOLATE=True: onnxruntime держит липкую CUDA-арену → отдельный процесс (gpu_align). AUTO=True:
авто-пост-шаг на КАЖДОЙ записи (предусловия нет — свой диапазон)."""
from __future__ import annotations

import json
from pathlib import Path

KEY = "forced"
LABEL = "Forced align (MMS)"
NOTE = ("MMS CTC forced align: точные границы по тексту аятов; ПОЛНОСТЬЮ независим — сам находит "
        "диапазон из своей акустики (ASR не нужен)")
SELECTABLE = True        # плоский независимый распознаватель (выбирается в форме, запускается сам)
AUTO = False             # больше НЕ авто-пост-шаг — равноправен с остальными (владелец)
ISOLATE = True
ALIGNED = True
PRIORITY = 20


def available() -> bool:
    try:
        import falign
        return falign.available()
    except Exception:
        return False


def run(rec, audio, quran, out_dir: Path, stage=None) -> dict:
    """Один GPU-проход MMS: эмиссии → свой диапазон (find_range) → force-align → возвраты. Зовётся
    из подпроцесса gpu_align (ISOLATE). `quran` — экземпляр Quran; `audio` — Path к аудио."""
    import falign
    import match_align

    if stage:
        stage("align")
    E, stride, wav, id2ch = falign.emissions(str(audio))
    # локализация диапазона из СВОЕЙ акустики MMS (романизованный скелет ↔ романизованный индекс)
    dec = falign.whole_decode_skeleton(E, stride, id2ch)
    index = match_align.build_romanized_index(quran)
    rng = match_align.find_range(None, quran, {}, {}, index=index, dec=dec, k=match_align._K_ROM)
    if not rng:
        raise RuntimeError("forced: не удалось определить диапазон из акустики MMS")
    verses = [(s, a, quran.surah(s).verses[a - 1].text) for s, a in rng]
    # force-align найденного диапазона по УЖЕ посчитанным эмиссиям (второй проход не нужен)
    sync_map = falign.align_verses(E, stride, wav, verses)
    meta = sync_map.setdefault("meta", {})
    meta["range_source"] = "forced-self"
    meta["range"] = f"{rng[0][0]}:{rng[0][1]}..{rng[-1][0]}:{rng[-1][1]}"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return sync_map
