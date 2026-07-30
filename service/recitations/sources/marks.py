"""Источник marks — WH «Мануал 2» (идея владельца tg_4547): облегчённая ручная разметка ТОЛЬКО
повторов. Диапазон аятов и структура повторов заданы человеком (файл `rec_dir/marks.json`,
записывает эндпоинт сохранения), тайминги ставит `w2v_align.forced_align(slots=)` — тот же надёжный
монотонный CTC-Viterbi, что раскладывает КАЖДОЕ звучание на свою копию (cov=1.0/fj=0). Отличие от
manual: не задаём тайминг каждого слова мышью/слухом, а лишь отмечаем что/сколько раз повторено.

Разметка (`marks.json`): {"verses": [[surah,ayah],...] в порядке чтения, "marks": [{start,end,count}]
по ПЛОСКИМ индексам слов диапазона}. Слоты собирает детерминированное ядро `pipeline.slots_from_marks`.

ISOLATE=True: w2v на GPU (липкая CUDA-арена) → отдельный процесс gpu_align. AUTO=False: гоняется НЕ
автоматически, а по явному сохранению разметки (эндпоинт). SELECTABLE — выбираемый прогон в плеере
наравне с forced/w2v. PRIORITY=6: человек структурировал повторы (близко к manual=5), но дефолтом НЕ
делаем автоматически (AUTO=False → появляется только когда владелец разметил)."""
from __future__ import annotations

import json
from pathlib import Path

KEY = "marks"
LABEL = "Ручная разметка повторов"
NOTE = ("облегчённый ручной элайнер (WH): человек отмечает ТОЛЬКО повторы (что/сколько раз), "
        "тайминги ставит forced_align(slots=) — надёжно, cov=1.0/fj=0")
SELECTABLE = True
AUTO = False
ISOLATE = True
ALIGNED = True
PRIORITY = 6
# требует РУЧНОЙ разметки (rec_dir/marks.json) → НЕ запускается при добавлении записи автоматически
# (форма добавления гонит только авто-запускаемые SELECTABLE: google/whisper). Появляется прогоном
# лишь когда владелец сохранил разметку через эндпоинт.
NEEDS_MARKUP = True


def available() -> bool:
    try:
        import w2v_align
        return w2v_align.available()
    except Exception:
        return False


def run(rec, audio, quran, out_dir: Path, stage=None) -> dict:
    """Разметка повторов → слоты → forced_align(slots=). Зовётся из подпроцесса gpu_align (ISOLATE).
    `marks.json` лежит рядом с аудио (rec_dir). `audio` — Path к аудио; `quran` — экземпляр Quran."""
    import w2v_align
    from recitations.pipeline import flat_range_words, slots_from_marks

    marks_path = Path(audio).parent / "marks.json"
    if not marks_path.is_file():
        raise RuntimeError(f"marks: нет разметки {marks_path} (сохрани через эндпоинт)")
    payload = json.loads(marks_path.read_text(encoding="utf-8"))
    verses = [(int(s), int(a)) for s, a in payload.get("verses", [])]
    marks = payload.get("marks", [])
    if not verses:
        raise RuntimeError("marks: пустой диапазон verses в разметке")

    flat = flat_range_words(quran, verses)
    slots = slots_from_marks(flat, marks)         # детерминированное ядро (валидирует пересечения/границы)

    if stage:
        stage("align")
    E, stride, idx2ch, ch2idx = w2v_align.emissions(str(audio))
    vtext = [(s, a, quran.surah(s).verses[a - 1].text) for s, a in verses]
    sync_map = w2v_align.forced_align(E, stride, vtext, idx2ch, ch2idx, str(audio), slots=slots)
    meta = sync_map.setdefault("meta", {})
    meta["range_source"] = "marks-manual"
    meta["range"] = f"{verses[0][0]}:{verses[0][1]}..{verses[-1][0]}:{verses[-1][1]}"
    meta["repeats_inserted"] = sum(1 for s in slots if s[4])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return sync_map
