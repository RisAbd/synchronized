"""Мост между Django-сервисом и ядром конвейера в `src/`.

Гоняет: ingest → распознавание (whisper локально | Google STT из кэша) → align → данные плеера.
Ядро (`src/`) остаётся тонким и импортируемым; здесь только оркестрация под сервис.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

from django.conf import settings

# подключаем ядро пайплайна
if str(settings.PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(settings.PIPELINE_SRC))


def _ensure_cudnn_path():
    """LD_LIBRARY_PATH на pip-путь cuDNN/cuBLAS (иначе faster-whisper падает)."""
    if "cudnn" in os.environ.get("LD_LIBRARY_PATH", ""):
        return
    import site
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        cudnn = Path(base) / "nvidia" / "cudnn" / "lib"
        cublas = Path(base) / "nvidia" / "cublas" / "lib"
        if cudnn.is_dir():
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn}:{cublas}:" + os.environ.get("LD_LIBRARY_PATH", "")
            return


@lru_cache(maxsize=1)
def _quran():
    from quran import Quran
    return Quran.load()


def _recognize(audio_path: Path, recognizer: str) -> dict:
    """Вернуть транскрипт {words:[{word,start,end}]}. Бэкенды: whisper | google(кэш)."""
    if recognizer == "google":
        # кэш ответов Google STT из старого проекта; ключ НЕ используется (только кэш).
        stem = audio_path.stem
        cache = Path(settings.GSTT_CACHE_DIR) / stem / "gstt_response.json"
        if not cache.is_file():
            raise FileNotFoundError(f"нет кэша Google STT: {cache}")
        return {"_gstt_path": str(cache)}  # align.load_transcript прочитает формат сам
    # whisper (по умолчанию)
    _ensure_cudnn_path()
    import asr
    return asr.transcribe(str(audio_path), language="ar")


def process(rec, on_stage=None) -> None:
    """Полный конвейер для записи Recitation. Мутирует и сохраняет объект."""
    def stage(name):
        rec.stage = name
        rec.save(update_fields=["stage", "updated_at"])
        if on_stage:
            on_stage(name)

    import ingest
    import align as align_mod
    from player import build_data

    work = Path(settings.WORK_DIR)

    # 1) ingest — получаем аудио
    stage("ingest")
    audio = ingest.fetch(rec.source_url, work)
    audio = Path(audio)

    # аудио для отдачи плееру кладём в AUDIO_DIR под стабильным именем
    audio_name = f"rec{rec.id}{audio.suffix}"
    dst = Path(settings.AUDIO_DIR) / audio_name
    if audio.resolve() != dst.resolve():
        shutil.copyfile(audio, dst)
    rec.audio_filename = audio_name
    rec.save(update_fields=["audio_filename", "updated_at"])

    # 2) распознавание
    stage("asr")
    tr = _recognize(dst, settings.RECOGNIZER)
    if "_gstt_path" in tr:
        words = align_mod.load_transcript(tr["_gstt_path"])
    else:
        tr_path = work / f"rec{rec.id}.transcript.json"
        tr_path.write_text(json.dumps(tr, ensure_ascii=False))
        words = align_mod.load_transcript(tr_path)

    # 3) align → sync-map
    stage("align")
    q = _quran()
    sync_map = align_mod.align(words, q)

    # 4) данные плеера
    stage("build")
    data = build_data(sync_map, q, audio_name)
    dur = data["timeline"][-1]["t"] if data["timeline"] else 0
    data["duration"] = round(dur)
    rec.data = data
    if not rec.title_ar:
        rec.title_ar = data["sections"][0]["title"] if data.get("sections") else ""
    rec.save(update_fields=["data", "title_ar", "updated_at"])
