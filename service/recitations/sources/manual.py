"""Источник manual — ручная привязка слов мышью (элайнер П12). Человек = истина, высший приоритет
авто-выбора в плеере. Создаётся ТОЛЬКО через `pipeline.build_manual_run` (из ручного элайнера),
поэтому SELECTABLE=False, AUTO=False, а `run()` через общий диспетчер не зовётся. Здесь — только
метаданные источника (KEY/LABEL/ALIGNED/PRIORITY), чтобы плеер/реестр знали про него."""
from __future__ import annotations

from pathlib import Path

KEY = "manual"
LABEL = "Ручной"
NOTE = ("ручная привязка слов мышью (элайнер П12); человек правит поверх forced/ASR — истина, "
        "высший приоритет авто-выбора")
SELECTABLE = False
AUTO = False
ISOLATE = False
ALIGNED = True
PRIORITY = 5             # выше всех: человек = истина


def available() -> bool:
    return True


def run(rec, audio, quran, out_dir: Path, stage=None) -> dict:
    raise NotImplementedError(
        "manual создаётся через pipeline.build_manual_run (из ручного элайнера), не через run()")
