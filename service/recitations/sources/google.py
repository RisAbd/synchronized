"""Источник google — Google Cloud STT (облако). Полностью независимый плагин: сам распознаёт
аудио, сам зовёт общий матчинг `match_align.align` (локализация + выравнивание слов), сам пишет
свои артефакты. Раньше жил веткой в pipeline._recognize + run_one — теперь самодостаточен.

Приоритет источника ответа: 1) свой raw.json (переиспользуем — не жжём квоту); 2) кэш ответов
старого проекта (по gstt_key/stem); 3) живой Google STT API (нужны ключ+бакет). Точнее whisper на
арабском, поэтому в авто-выборе плеера стоит выше whisper (PRIORITY)."""
from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings

KEY = "google"
LABEL = "Google STT"
NOTE = "точнее на арабском; из кэша ответов или живого API"
SELECTABLE = True
AUTO = False
ISOLATE = False
ALIGNED = False          # сырой ASR (не выравнивание к известному тексту)
PRIORITY = 30


def available() -> bool:
    # источник ответа определяется в run() (свой raw / кэш / живой API) — на уровне модуля
    # ничего блокирующего нет; невозможность распознать всплывёт понятной ошибкой в run().
    return True


def run(rec, audio, quran, out_dir: Path, stage=None) -> dict:
    import match_align
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.json"
    tr_path = out_dir / "transcript.json"

    if stage:
        stage("asr")
    # 1) уже распознавали этот прогон (raw.json есть) — переиспользуем, чтобы не жечь квоту.
    # 2) иначе кэш ответов из старого проекта (ключ = gstt_key записи или stem аудио) — бесплатно.
    # 3) иначе живой Google STT API (если задан ключ+бакет) — сохраняем ответ в raw.json.
    if raw_path.is_file() and "results" in json.loads(raw_path.read_text() or "{}"):
        src = raw_path
    else:
        key = rec.gstt_key or audio.stem
        cache = Path(settings.GSTT_CACHE_DIR) / key / "gstt_response.json"
        if cache.is_file():
            raw = json.loads(cache.read_text())
            raw_path.write_text(json.dumps(raw, ensure_ascii=False))
        else:
            import gstt
            if not (settings.GSTT_LIVE and gstt.is_available()):
                raise FileNotFoundError(
                    f"нет кэша Google STT для '{key}' ({cache}), а живой API выключен "
                    f"(нужны env GOOGLE_APPLICATION_CREDENTIALS + SYNC_GSTT_BUCKET, "
                    f"SYNC_GSTT_LIVE≠0).")
            resp = gstt.recognize(audio, bucket_name=settings.GSTT_BUCKET)
            raw_path.write_text(json.dumps(resp, ensure_ascii=False, indent=2))
        src = raw_path
    words = match_align.load_transcript(src)

    # нормализованный вход матчинга — для дебага (что реально скормили)
    tr_path.write_text(json.dumps(
        [{"word": w.word, "start": w.start, "end": w.end, "norm": w.norm} for w in words],
        ensure_ascii=False, indent=2))

    if stage:
        stage("align")
    sync_map = match_align.align(words, quran)
    # счётчики ASR↔эталон (идея quran-align): hits/subs/ins/dels/wer против текста найденного диапазона
    sync_map.setdefault("meta", {})["match"] = match_align.match_stats(
        [w.norm for w in words], sync_map, quran)
    (out_dir / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    return sync_map
