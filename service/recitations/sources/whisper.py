"""Источник whisper — локальный Whisper (Tarteel, дообучен под коранический арабский) на GPU.
Самодостаточный плагин: транскрибирует аудио (ct2), зовёт общий матчинг `match_align.align`, пишет
артефакты. Сырой ответ со словами сохраняем в raw.json и переиспользуем (не жжём GPU повторно —
симметрично google). Модель — SYNC_WHISPER_MODEL (по умолчанию ct2-tarteel-base)."""
from __future__ import annotations

import json
from pathlib import Path

KEY = "whisper"
LABEL = "Whisper (Tarteel)"
NOTE = "локально на GPU; tarteel-ai/whisper-base-ar-quran — дообучена под коранический арабский"
SELECTABLE = True
AUTO = False
ISOLATE = False
ALIGNED = False          # сырой ASR
PRIORITY = 40


def available() -> bool:
    return True


def run(rec, audio, quran, out_dir: Path, stage=None) -> dict:
    import match_align
    from recitations import pipeline
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.json"
    tr_path = out_dir / "transcript.json"

    if stage:
        stage("asr")
    # уже распознавали (raw.json со словами) — переиспользуем, НЕ жжём GPU заново; иначе живой whisper.
    if not (raw_path.is_file() and json.loads(raw_path.read_text() or "{}").get("words")):
        pipeline._ensure_cudnn_path()
        import asr
        raw = asr.transcribe(str(audio), language="ar")
        raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    words = match_align.load_transcript(raw_path)

    tr_path.write_text(json.dumps(
        [{"word": w.word, "start": w.start, "end": w.end, "norm": w.norm} for w in words],
        ensure_ascii=False, indent=2))

    if stage:
        stage("align")
    sync_map = match_align.align(words, quran)
    sync_map.setdefault("meta", {})["match"] = match_align.match_stats(
        [w.norm for w in words], sync_map, quran)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return sync_map
