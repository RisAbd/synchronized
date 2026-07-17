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
    src = pipeline._forced_source(rec)
    if src is None:
        print("нет готового ASR-прогона (google/whisper) для диапазона аятов", file=sys.stderr)
        return 3
    verses = falign.verses_from_data(src.data)
    if not verses:
        print(f"в прогоне-источнике '{src.recognizer}' нет разделов/аятов", file=sys.stderr)
        return 4

    if recognizer == rz.W2V:
        import w2v_align
        # окна аятов из timeline ASR-источника (для нарезки длинного аудио на короткие сегменты)
        tl = (src.data.get("timeline") or [])
        tmap = {(t["surah"], t["ayah"]): t["t"] for t in tl}
        windows = [[tmap.get((s, a)), None] for s, a, _ in verses]
        sync_map = w2v_align.align(str(audio), verses, windows=windows)
    else:
        sync_map = falign.align(str(audio), verses)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
