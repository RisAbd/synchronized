"""Источник w2v-ORACLE — тот же wav2vec2, но возвраты чтеца через ОРАКУЛ правдоподобия (WG,
план владельца tg_4539), а не repeat-aware Viterbi. greedy_repeat_slots авто-генерит расширенный
текст повторов (локальный CTC path_score H0×m vs H1×m+1 судит каждую копию, порогов нет) →
forced_align(slots=) монотонно раскладывает КАЖДОЕ звучание на свою копию. Берёт фразы-перечитки,
которые акустический Viterbi теряет (السماوات والأرض, لا إله إلا هو, ذلكم الله — модель путает буквы).

Отдельный ВЫБИРАЕМЫЙ прогон рядом с дефолтным `w2v` — чтобы владелец сравнил на слух, не меняя
задеплоенное вслепую (rep-точность валидируется только ухом). PRIORITY=15 > w2v(10) → дефолтом НЕ
становится (при равном fj=0 побеждает меньший PRIORITY). AUTO=False — не гоняется автоматически (не
дублирует GPU-эмиссии); заполняется офлайн по кэшу эмиссий скриптом (как test2). Идентичен w2v.run,
только режим возвратов = oracle."""
from __future__ import annotations

import json
from pathlib import Path

KEY = "w2vo"
LABEL = "Forced align (wav2vec2, оракул повторов)"
NOTE = ("wav2vec2 + оракул правдоподобия для повторов (авто-структура перечиток через forced_align "
        "slots): берёт фразы-перечитки, что акустический Viterbi теряет; выбираемый для сравнения на слух")
SELECTABLE = True        # плоский независимый распознаватель (выбирается в форме, запускается сам)
AUTO = False
ISOLATE = True
ALIGNED = True
PRIORITY = 15            # > w2v(10) → дефолтом не становится; < forced(20)


def available() -> bool:
    try:
        import w2v_align
        return w2v_align.available()
    except Exception:
        return False


def run(rec, audio, quran, out_dir: Path, stage=None) -> dict:
    """Один GPU-проход: эмиссии → диапазон (find_segments) → авто-структура повторов (оракул) →
    forced_align(slots=). Зовётся из подпроцесса gpu_align (ISOLATE)."""
    import match_align
    import w2v_align

    from . import w2v as _w2v          # общий срез краёв не-Корана (_edge_trim/_cap_tail)

    E, stride, idx2ch, ch2idx = w2v_align.emissions(str(audio))
    index = match_align.build_index(quran)
    spans = match_align.find_segments(E, quran, idx2ch, ch2idx, index=index, return_spans=True)
    if not spans:
        raise RuntimeError("w2vo: не удалось определить диапазон из акустики")
    rng = [sa for sp in spans for sa in sp["seg"]]
    verses = [(s, a, quran.surah(s).verses[a - 1].text) for s, a in rng]
    import os
    # НАМАЗ (мультисегмент + корановская модель): пооконный align — как в w2v, чтобы w2vo и w2v были
    # согласованы при сравнении на слух (иначе w2vo размазывает Фатиху-2 по такбиру). Повторы ВНУТРИ
    # намазового куска редки (Фатиха не перечитывается) → slots-оракул тут не нужен. Одиночные реки
    # (перечитки чтеца) идут ОРАКУЛ-путём slots (ради чего w2vo и существует).
    perwin = (len(spans) >= 2 and os.environ.get("SYNC_W2V_MODEL")
              and os.environ.get("SYNC_W2V_PERWIN", "1") != "0")
    sync_map = None
    if perwin:
        sync_map = _w2v._perwindow_align(E, spans, quran, stride, idx2ch, ch2idx, str(audio), index)
    if sync_map is not None:
        meta = sync_map.setdefault("meta", {})
        meta["repeats_mode"] = "perwindow-forced"
        meta["windows"] = len(spans)
    else:
        slots = w2v_align.greedy_repeat_slots(E, verses, ch2idx, stride)   # слоты повторов — по исходному E
        E_al, edge = _w2v._edge_trim(E, spans, stride, idx2ch, ch2idx)     # forced_align — по маскированному
        sync_map = w2v_align.forced_align(E_al, stride, verses, idx2ch, ch2idx, str(audio), slots=slots)
        if edge:
            _w2v._cap_tail(sync_map, edge["window"][1])
        meta = sync_map.setdefault("meta", {})
        meta["repeats_mode"] = "oracle-greedy-forced"
        meta["repeats_inserted"] = sum(1 for s in slots if s[4])
        if edge:
            meta["edge_trim"] = edge
    meta["range_source"] = "w2v-self"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return sync_map
