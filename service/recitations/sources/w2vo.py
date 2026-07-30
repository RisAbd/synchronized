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

    E, stride, idx2ch, ch2idx = w2v_align.emissions(str(audio))
    index = match_align.build_index(quran)
    rng = match_align.find_segments(E, quran, idx2ch, ch2idx, index=index)
    if not rng:
        raise RuntimeError("w2vo: не удалось определить диапазон из акустики")
    verses = [(s, a, quran.surah(s).verses[a - 1].text) for s, a in rng]
    slots = w2v_align.greedy_repeat_slots(E, verses, ch2idx, stride)
    sync_map = w2v_align.forced_align(E, stride, verses, idx2ch, ch2idx, str(audio), slots=slots)
    meta = sync_map.setdefault("meta", {})
    meta["range_source"] = "w2v-self"
    meta["repeats_mode"] = "oracle-greedy-forced"
    meta["repeats_inserted"] = sum(1 for s in slots if s[4])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return sync_map
