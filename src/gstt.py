"""Живой Google Speech-to-Text для арабской рецитации.

Google требует async-распознавание по gs:// URI для аудио длиннее ~1 мин, поэтому
файл (перекодированный в FLAC 16k mono — рекомендованный Google путь, снимает догадки
про encoding/sample-rate) заливается во временный объект бакета, затем
`long_running_recognize`. Ответ возвращается в ТОМ ЖЕ виде, что кэш `gstt_response.json`
({"results": [...]}), — его понимает `align.load_transcript`.

⚠️ Ключ сервис-аккаунта — ТОЛЬКО через env `GOOGLE_APPLICATION_CREDENTIALS`, в репо не
коммитим (см. .gitignore). Бакет — env `SYNC_GSTT_BUCKET`.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


def is_available() -> bool:
    """Живой Google STT возможен, если задан путь к ключу сервис-аккаунта и бакет."""
    cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    return bool(cred and Path(cred).is_file() and os.environ.get("SYNC_GSTT_BUCKET"))


def _to_flac16k(src: Path, dst: Path) -> None:
    """Перекодировать в FLAC 16 кГц моно — канонический вход Google STT."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg перекодировка в FLAC не удалась: {proc.stderr.decode()[-500:]}")


def recognize(audio_path, *, bucket_name: str, language: str = "ar",
              blob_prefix: str = "synchronized/", timeout: int = 1800) -> dict:
    """Распознать аудио живым Google STT, вернуть ответ в формате gstt_response.json.

    Заливает FLAC во временный blob, гоняет long_running_recognize, удаляет blob.
    """
    import proto
    from google.cloud import speech, storage

    audio_path = Path(audio_path)
    with tempfile.TemporaryDirectory() as tmp:
        flac = Path(tmp) / (audio_path.stem + ".flac")
        _to_flac16k(audio_path, flac)

        st = storage.Client()
        bucket = st.bucket(bucket_name)
        blob = bucket.blob(f"{blob_prefix}{flac.name}")
        blob.upload_from_filename(str(flac))
        gcs_uri = f"gs://{bucket_name}/{blob.name}"

        try:
            client = speech.SpeechClient()
            # модель НЕ задаём: для арабского спец-модели (latest_long и т.п.) не поддержаны
            # ("model not supported for language: ar") — используем дефолтную.
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
                sample_rate_hertz=16000,
                language_code=language,
                enable_word_time_offsets=True,
                enable_automatic_punctuation=False,
            )
            audio = speech.RecognitionAudio(uri=gcs_uri)
            operation = client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=timeout)
            return json.loads(proto.Message.to_json(response))
        finally:
            try:
                blob.delete()
            except Exception:
                pass  # мусор в бакете не критичен, распознавание уже сделано
