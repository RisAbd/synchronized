"""Источник w2v — wav2vec2 + СВОЙ CTC-Viterbi (без whisperx). ПОЛНОСТЬЮ независим (директива
владельца 24.07): audio → своя акустика (эмиссии) → сам находит диапазон аятов (`match_align.
find_range`) → выравнивание диапазона С ВОЗВРАТАМИ за один проход (`w2v_align.repeat_align` —
repeat-aware Viterbi, ход назад по акустике). ASR ему НЕ нужен, данные других источников НЕ
наследует. Держит слово сквозь мадд → честный coverage; дефолтный выравниватель (PRIORITY выше forced).

ISOLATE=True: гоняется в отдельном процессе (gpu_align) — torch(w2v) и onnxruntime(forced) в одном
процессе на 6ГБ = OOM (липкая CUDA-арена). AUTO=True: авто-пост-шаг на каждой записи."""
from __future__ import annotations

import json
import os
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


def _align_with_repeats(E, stride, verses, idx2ch, ch2idx, audio_path):
    """Возвраты чтеца двумя режимами (env `SYNC_W2V_REPEATS`):

    • `viterbi` (по умолчанию) — ОДИН проход repeat-aware CTC-Viterbi: аллайнер сам ходит назад по
      акустике (см. w2v_align.repeat_align). Ни пред-детекта, ни порогов — только окно R и штраф P=0.
      Ловит длинно-span'овые возвраты, но фразы-перечитки, где модель путает буквы (بديع: ب→ق), теряет.

    • `oracle` — авто-структура повторов через ОРАКУЛ правдоподобия (WG, план владельца tg_4539):
      greedy_repeat_slots судит по локальному CTC path_score H0(×m) vs H1(×m+1), генерит расширенный
      текст повторов → forced_align(slots=) монотонно раскладывает КАЖДОЕ звучание на свою копию. Берёт
      фразы-перечитки, которые Viterbi-режим теряет (السماوات والأرض, لا إله إلا هو). Порогов «сколько
      повторов» нет — решает модель. Требует валидации слухом на каждой реке (rep-точность — только ухо).

    Возвращает (sync_map, rep_info)."""
    import w2v_align
    mode = (os.environ.get("SYNC_W2V_REPEATS", "viterbi") or "viterbi").lower()
    if mode == "oracle":
        slots = w2v_align.greedy_repeat_slots(E, verses, ch2idx, stride)
        sync = w2v_align.forced_align(E, stride, verses, idx2ch, ch2idx, audio_path, slots=slots)
        return sync, {"repeats_mode": "oracle-greedy-forced",
                      "repeats_inserted": sum(1 for s in slots if s[4])}
    sync = w2v_align.repeat_align(E, stride, verses, idx2ch, ch2idx, audio_path)
    meta = sync.get("meta") or {}
    return sync, {"repeats_mode": "1pass-repeat-viterbi",
                  "repeats_inserted": meta.get("reps", 0),
                  "repeat_R": meta.get("repeat_R"), "repeat_P": meta.get("repeat_P")}


def _fmt_segments(rng) -> str:
    """компактная запись сегментов: разрывы по номерам аятов → отдельные куски (1:1-1:7, 17:1-17:60)."""
    parts, s0 = [], 0
    for i in range(1, len(rng) + 1):
        brk = (i == len(rng)) or rng[i][0] != rng[i - 1][0] or rng[i][1] != rng[i - 1][1] + 1
        if brk:
            a, b = rng[s0], rng[i - 1]
            parts.append(f"{a[0]}:{a[1]}-{b[0]}:{b[1]}")
            s0 = i
    return ", ".join(parts)


def _seg_count(rng) -> int:
    n = 1 if rng else 0
    for i in range(1, len(rng)):
        if rng[i][0] != rng[i - 1][0] or rng[i][1] != rng[i - 1][1] + 1:
            n += 1
    return n


def run(rec, audio, quran, out_dir: Path, stage=None) -> dict:
    """Один GPU-проход: эмиссии → диапазон → выравнивание с возвратами (repeat-aware Viterbi).
    Зовётся из подпроцесса gpu_align (ISOLATE). `quran` — экземпляр Quran; `audio` — Path к аудио."""
    import match_align
    import w2v_align

    E, stride, idx2ch, ch2idx = w2v_align.emissions(str(audio))
    index = match_align.build_index(quran)
    # МУЛЬТИ-СЕГМЕНТ (владелец 26.07): аудио звучит в РАЗНЫХ местах Корана (Фатиха + основная сура +
    # такбиры, записи с намаза) — общий find_segments находит ВСЕ читаемые места, не один непрерывный.
    rng = match_align.find_segments(E, quran, idx2ch, ch2idx, index=index)   # [(surah, ayah), ...] в порядке чтения
    if not rng:
        raise RuntimeError("w2v: не удалось определить диапазон из акустики")
    verses = [(s, a, quran.surah(s).verses[a - 1].text) for s, a in rng]
    sync_map, rep_info = _align_with_repeats(E, stride, verses, idx2ch, ch2idx, str(audio))
    meta = sync_map.setdefault("meta", {})
    meta["range_source"] = "w2v-self"
    meta["range"] = _fmt_segments(rng)
    meta["segments"] = _seg_count(rng)
    meta.update(rep_info)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return sync_map
