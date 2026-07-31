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
SELECTABLE = True        # плоский независимый распознаватель: выбирается в форме, запускается сам по себе
AUTO = False             # больше НЕ авто-пост-шаг (нет «особых» источников — все равноправны, владелец)
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


def _edge_trim(E, spans, stride, idx2ch, ch2idx):
    """Срез КРАЁВ не-Коран времени (костяк владельца: не маппить такбир/саламы на текст).

    Проблема: repeat_align — ОДИН монотонный CTC-Viterbi по ВСЕЙ эмиссии [0,T] → открывающий такбир
    и закрывающие саламы/дуа поглощаются растянутыми крайними словами («первая Фатиха мапится на шум»,
    «последний аят на саламы» — жалобы владельца по rec16-намазу). Фикс: форсируем blank на кадрах ДО
    первого и ПОСЛЕ последнего найденного сегмента → там подсветки нет, слова сжимаются в реальное окно
    рецитации. Границы берём из find_segments (pos первого / pos_end последнего доминирующего сегмента —
    надёжные якоря). Внутренние такбиры между кусками пока НЕ трогаем (их концы у плохо декодированных
    мелодичных кусков ненадёжны — отдельный шаг). Гард EDGE_MIN: срезаем только КРУПНЫЙ краевой не-Коран
    (намаз), у чистых рек с малыми краями (<EDGE_MIN) ничего не масикруем → они байт-в-байт как были."""
    import numpy as np
    if os.environ.get("SYNC_W2V_EDGE_TRIM", "1") == "0" or not spans:
        return E, None
    # только МУЛЬТИСЕГМЕНТ (намаз: Фатиха+сура+такбиры) — там края = такбир/саламы. Чистые односегментные
    # реки (одна сура подряд) НЕ трогаем: их края — тишина, текущее поведение владелец считает нормой.
    if len(spans) < 2 and os.environ.get("SYNC_W2V_EDGE_FORCE", "0") != "1":
        return E, None
    EDGE_MIN = float(os.environ.get("SYNC_W2V_EDGE_MIN", "10") or 10)     # с: минимум краевого не-Корана
    MARGIN = float(os.environ.get("SYNC_W2V_EDGE_MARGIN", "1.5") or 1.5)  # с: запас, чтоб не клипать слово
    special = {ch2idx.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>", "|", "-", "ـ")} - {None}
    import match_align
    _, ctimes = match_align.greedy_skeleton(E, idx2ch, special, times=True, stride_ms=stride)
    N = len(ctimes)
    if N == 0:
        return E, None
    p0 = max(0, min(spans[0]["pos"], N - 1))
    p1 = max(0, min(spans[-1]["pos_end"], N - 1))
    t0, t1 = ctimes[p0], ctimes[p1]
    dur = E.shape[0] * stride / 1000.0
    blank = ch2idx.get("<pad>")
    if blank is None:
        return E, None
    lead = t0 > EDGE_MIN
    trail = (dur - t1) > EDGE_MIN
    if not (lead or trail):
        return E, None
    E2 = E.copy()
    f0 = f1 = None
    if lead:
        f0 = max(0, int((t0 - MARGIN) * 1000.0 / stride))
        E2[:f0, :] = -1e9; E2[:f0, blank] = 0.0     # форс blank на вступительном такбире
    if trail:
        f1 = min(E.shape[0], int((t1 + MARGIN) * 1000.0 / stride))
        E2[f1:, :] = -1e9; E2[f1:, blank] = 0.0     # форс blank на хвостовых саламах/дуа
    info = {"window": [round(t0, 1), round(t1, 1)], "audio": round(dur, 1),
            "lead_trimmed": round(t0, 1) if lead else 0.0,
            "trail_trimmed": round(dur - t1, 1) if trail else 0.0}
    return E2, info


def _cap_tail(sync_map, t_end_max: float):
    """Обрезать хвост: слова после конца окна рецитации (t ≥ t_end_max) — убрать, у остальных t_end
    не длиннее окна. Последнее слово в аллайнере «держится» до конца аудио явно (не по кадрам) → без
    этого оно подсвечивалось бы весь хвостовой салам/дуа. Работает по word_timeline/timeline/char."""
    for key in ("word_timeline", "timeline", "char_timeline"):
        arr = sync_map.get(key)
        if not arr:
            continue
        kept = []
        for w in arr:
            if w.get("t", 0.0) >= t_end_max:
                continue                                  # начало уже за окном → полностью в саламе
            if w.get("t_end") is not None and w["t_end"] > t_end_max:
                w["t_end"] = t_end_max
            kept.append(w)
        sync_map[key] = kept


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
    spans = match_align.find_segments(E, quran, idx2ch, ch2idx, index=index, return_spans=True)
    if not spans:
        raise RuntimeError("w2v: не удалось определить диапазон из акустики")
    rng = [sa for sp in spans for sa in sp["seg"]]           # плоский список аятов в порядке чтения
    verses = [(s, a, quran.surah(s).verses[a - 1].text) for s, a in rng]
    # срез краёв не-Коран (такбир/саламы) → слова не растягиваются на них (см. _edge_trim)
    E_al, edge = _edge_trim(E, spans, stride, idx2ch, ch2idx)
    sync_map, rep_info = _align_with_repeats(E_al, stride, verses, idx2ch, ch2idx, str(audio))
    if edge:
        _cap_tail(sync_map, edge["window"][1])   # последнее слово «держится» до конца аудио явно — обрезать до окна
    meta = sync_map.setdefault("meta", {})
    meta["range_source"] = "w2v-self"
    meta["range"] = _fmt_segments(rng)
    meta["segments"] = _seg_count(rng)
    if edge:
        meta["edge_trim"] = edge
    meta.update(rep_info)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return sync_map
