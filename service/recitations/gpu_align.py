"""Запуск GPU-источника (ISOLATE) в ОТДЕЛЬНОМ короткоживущем процессе — GPU-изоляция.

Зачем отдельный процесс. Карта 6ГБ, а на шаге выравнивания сталкиваются два фреймворка:
onnxruntime-gpu (MMS forced) держит ЛИПКУЮ CUDA-арену — не отдаёт VRAM в пределах процесса даже
после удаления сессии; torch (w2v) в том же процессе → CUDA OutOfMemory. Плюс до этого в процессе
воркера мог остаться резидентный ct2-whisper. Решение: gpu-источник гоняем как подпроцесс —
загрузил фреймворк → выровнял → записал sync-map.json → вышел, и ОС освобождает всю VRAM. Родитель
(celery-воркер) на этом шаге GPU-фреймворки сам не грузит.

Обобщён (директива владельца: плоские плагины) — не ветвится по типу источника: динамически берёт
модуль-источник из пакета `sources` по ключу и зовёт его единый `run()`.

Запуск: python -m recitations.gpu_align <rec_id> <source_key> <out_dir>
Результат: <out_dir>/sync-map.json (код возврата 0). Ошибку печатаем в stderr, код != 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: python -m recitations.gpu_align <rec_id> <source_key> <out_dir>", file=sys.stderr)
        return 2
    rec_id = int(sys.argv[1])
    key = sys.argv[2]
    out_dir = Path(sys.argv[3])

    import os
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/app/service")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "synchronized.settings")
    import django
    django.setup()

    from recitations.models import Recitation
    from recitations import pipeline, sources
    from quran import Quran

    mod = sources.get(key)
    if mod is None:
        print(f"неизвестный источник: {key!r}", file=sys.stderr)
        return 2

    rec = Recitation.objects.get(pk=rec_id)
    audio = pipeline.ensure_audio(rec)
    try:
        sync_map = mod.run(rec, Path(audio), Quran.load(), out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"{key}: {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    # run() уже записал sync-map.json; на всякий случай гарантируем файл из возвращённого dict
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "sync-map.json").exists():
        (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
