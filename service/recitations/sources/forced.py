"""Источник forced — MMS CTC forced alignment по известному тексту аятов. Точные границы: берёт
диапазон читаемых аятов из готового ASR-прогона (google/whisper) и выравнивает их текст к аудио
(MMS). Возвраты чтеца (П8) детектит свой `falign._detect_repeats` по MMS-эмиссиям.

ISOLATE=True: onnxruntime держит липкую CUDA-арену → отдельный процесс (gpu_align). AUTO=True:
авто-пост-шаг. ready(): нужен готовый ASR-источник диапазона (пока свой find_range не подключён —
это инкремент 3, снятие _forced_source)."""
from __future__ import annotations

import json
from pathlib import Path

KEY = "forced"
LABEL = "Forced align (MMS)"
NOTE = "точные границы: выравнивает текст аятов к аудио (MMS CTC); нужен готовый ASR-прогон для диапазона"
SELECTABLE = False
AUTO = True
ISOLATE = True
ALIGNED = True
PRIORITY = 20


def available() -> bool:
    try:
        import falign
        return falign.available()
    except Exception:
        return False


def ready(rec) -> bool:
    """Нужен готовый ASR-прогон (из него берётся диапазон аятов). Без него авто-запуск пропускаем
    тихо (не заводим ERROR-прогон)."""
    from recitations import pipeline
    return pipeline._forced_source(rec) is not None


def run(rec, audio, quran, out_dir: Path, stage=None) -> dict:
    import falign
    from recitations import pipeline

    src = pipeline._forced_source(rec)
    if src is None:
        raise RuntimeError(
            "нет готового прогона (google/whisper) для диапазона аятов — сначала распознайте "
            "запись каким-нибудь ASR, затем добавьте выравнивание")
    verses = falign.verses_from_data(src.data)
    if not verses:
        raise RuntimeError(f"в прогоне-источнике '{src.recognizer}' нет разделов/аятов")
    if stage:
        stage("align")
    sync_map = falign.align(str(audio), verses)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return sync_map
