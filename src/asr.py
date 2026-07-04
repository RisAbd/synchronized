"""M3 — транскрипция аудио (речь → слова + тайминги) локальным faster-whisper.

Вход: путь к аудио (любой формат — faster-whisper сам декодирует и ресемплит в 16k mono).
Выход: transcript.json — {audio, language, words:[{word, start, end, prob}]}.
Формат совместим с align.load_transcript.

ВАЖНО (cuDNN): ctranslate2 не находит cuDNN по умолчанию. Перед запуском выставить:
  export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:\
$HOME/.local/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
(иначе `Unable to load libcudnn_ops.so.9`, exit 134). См. skill audio-task.

CLI:  python3 asr.py <audio> [out.json] [lang]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _default_model() -> str:
    """Какую whisper-модель грузить. env SYNC_WHISPER_MODEL — либо имя размера (large-v3, small…),
    либо ПУТЬ к каталогу CTranslate2 (напр. дообученная под коранический арабский
    tarteel-ai/whisper-base-ar-quran, сконвертированная `ct2-transformers-converter`).
    Дефолт large-v3 — ванильный, измеренно ПЛОХ на арабском (rec1 6 слов на 21 мин), поэтому
    в докер-воркере переопределяем на ct2-tarteel (см. docker-compose + ./cache/ct2-tarteel-base)."""
    import os
    return os.environ.get("SYNC_WHISPER_MODEL", "").strip() or "large-v3"


def _load_model(model_size: str):
    """Грузим модель. По умолчанию — ТОЛЬКО GPU (cuda): на CPU whisper на арабском бесполезен
    (даёт кашу, напр. 5 слов на 20 мин — см. владелец 04.07), и тратить на него время нельзя.
    docker-воркер теперь ходит на GPU (nvidia-container-toolkit + LD_LIBRARY_PATH, см. compose),
    так что whisper крутится прямо в сервисе. Если GPU нет — whisper падает, распознаёт google.
    SYNC_ASR_DEVICE=cpu — явный опт-ин на CPU (если кто-то реально хочет), cuda — то же, что дефолт."""
    import os
    from faster_whisper import WhisperModel

    dev = os.environ.get("SYNC_ASR_DEVICE", "").lower()
    if dev == "cpu":
        attempts = [("cpu", "int8")]
    else:  # дефолт и 'cuda' — только GPU, без CPU-фолбэка
        attempts = [("cuda", "float16"), ("cuda", "int8_float16")]
    last = None
    for device, compute in attempts:
        try:
            return WhisperModel(model_size, device=device, compute_type=compute)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"whisper {device}/{compute} не вышло ({e})", file=sys.stderr)
    raise RuntimeError(
        f"whisper: нет GPU (cuda) — распознавание пропущено, CPU-фолбэк отключён намеренно "
        f"(бесполезен на арабском; SYNC_ASR_DEVICE=cpu чтобы форсить). Последняя ошибка: {last}")


def _use_vad() -> bool:
    """VAD (silero) по умолчанию ВЫКЛЮЧЕН. На мелодичном таджвиде он катастрофически ошибается —
    принимает распевное чтение за тишину и выкидывает ~97% аудио (измерено: rec9 18 мин →
    VAD оставил 34с; с VAD 5 слов, без VAD 896 — сопоставимо с google 915). Включить: SYNC_ASR_VAD=1."""
    import os
    return os.environ.get("SYNC_ASR_VAD", "").strip().lower() in ("1", "true", "yes", "on")


def transcribe(audio: str | Path, language: str = "ar", model_size: str | None = None) -> dict:
    audio = str(audio)
    model = _load_model(model_size or _default_model())

    segments, info = model.transcribe(
        audio,
        language=language,
        word_timestamps=True,
        vad_filter=_use_vad(),
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5,
    )

    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({
                "word": w.word.strip(),
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "prob": round(w.probability, 3),
            })
        # прогресс в stderr
        print(f"[{int(seg.start//60):02d}:{int(seg.start%60):02d}] {seg.text.strip()}",
              file=sys.stderr, flush=True)

    return {
        "audio": audio,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 1),
        "words": [w for w in words if w["word"]],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 asr.py <audio> [out.json] [lang]", file=sys.stderr)
        sys.exit(1)
    audio = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "work/transcript.json"
    lang = sys.argv[3] if len(sys.argv) > 3 else "ar"

    result = transcribe(audio, language=lang)
    Path(out).parent.mkdir(exist_ok=True, parents=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nслов: {len(result['words'])}, lang={result['language']} -> {out}", file=sys.stderr)
