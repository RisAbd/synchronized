"""Выравнивание (forced MMS / wav2vec2) в ОТДЕЛЬНОМ короткоживущем процессе — GPU-изоляция.

Зачем отдельный процесс. Карта 6ГБ, а на шаге выравнивания сталкиваются два фреймворка:
onnxruntime-gpu (MMS forced) держит ЛИПКУЮ CUDA-арену — не отдаёт VRAM в пределах процесса даже
после удаления сессии; torch (w2v/whisperx) в том же процессе → CUDA OutOfMemory. Плюс до этого
в процессе воркера мог остаться резидентный ct2-whisper. Решение: каждый выравниватель гоняем
как подпроцесс — загрузил фреймворк → выровнял → записал sync-map.json → вышел, и ОС освобождает
всю VRAM. Родитель (celery-воркер) на этом шаге GPU-фреймворки сам не грузит.

Запуск: python -m recitations.gpu_align <rec_id> <recognizer> <out_dir>
Результат: <out_dir>/sync-map.json (код возврата 0). Ошибку печатаем в stderr, код != 0.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: python -m recitations.gpu_align <rec_id> <recognizer> <out_dir>", file=sys.stderr)
        return 2
    rec_id = int(sys.argv[1])
    recognizer = sys.argv[2]
    out_dir = Path(sys.argv[3])

    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/app/service")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "synchronized.settings")
    import django
    django.setup()

    from recitations.models import Recitation
    from recitations import pipeline, recognizers as rz
    import falign

    rec = Recitation.objects.get(pk=rec_id)
    audio = pipeline.ensure_audio(rec)

    if recognizer == rz.W2V:
        # ПОЛНАЯ независимость (директива владельца 24.07): w2v НЕ берёт диапазон/окна у ASR.
        # Своя акустика: эмиссии → find_range (какие аяты) → force-align этого диапазона → sync_map.
        import w2v_align
        import w2v_range
        from quran import Quran
        E, stride, idx2ch, ch2idx = w2v_align.emissions(str(audio))
        q = Quran.load()
        index = w2v_range.build_index(q)
        rng = w2v_range.find_range(E, q, idx2ch, ch2idx, index=index)   # [(surah, ayah), ...]
        if not rng:
            print("w2v: не удалось определить диапазон из акустики", file=sys.stderr)
            return 3
        verses = [(s, a, q.surah(s).verses[a - 1].text) for s, a in rng]
        # грубые старты аятов для нарезки длинного аудио — из СВОЕЙ акустики (k-грамм-попадания
        # decode-время→аят), НЕ из ASR-таймлайна. _fill_starts добьёт пропуски интерполяцией.
        starts = w2v_range.ayah_start_hints(E, verses, index, idx2ch, ch2idx, stride)
        windows = [[st, None] for st in starts]
        sync_map = w2v_align.align(str(audio), verses, windows=windows)
        # возвраты/перечитки чтеца (П8) из СВОЕЙ акустики w2v (не наследуем у forced) — вклейка
        # rep-точек в word_timeline, как в falign.align. Эмиссии переиспользуем (посчитаны выше).
        import w2v_repeats
        reps = w2v_repeats.detect(E, stride, idx2ch, ch2idx,
                                  sync_map["word_timeline"], verses, str(audio))
        if reps:
            sync_map["word_timeline"].extend(reps)
            sync_map["word_timeline"].sort(key=lambda w: (w["t"], w["surah"], w["ayah"], w["wi"]))
        meta = sync_map.setdefault("meta", {})
        meta["range_source"] = "w2v-self"
        meta["range"] = f"{rng[0][0]}:{rng[0][1]}..{rng[-1][0]}:{rng[-1][1]}"
        meta["repeats_inserted"] = len(reps)
    else:
        src = pipeline._forced_source(rec)
        if src is None:
            print("нет готового ASR-прогона (google/whisper) для диапазона аятов", file=sys.stderr)
            return 3
        verses = falign.verses_from_data(src.data)
        if not verses:
            print(f"в прогоне-источнике '{src.recognizer}' нет разделов/аятов", file=sys.stderr)
            return 4
        sync_map = falign.align(str(audio), verses)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
