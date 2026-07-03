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


def transcribe(audio: str | Path, language: str = "ar", model_size: str = "large-v3") -> dict:
    from faster_whisper import WhisperModel

    audio = str(audio)
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"float16 failed ({e}); fallback int8_float16", file=sys.stderr)
        model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

    segments, info = model.transcribe(
        audio,
        language=language,
        word_timestamps=True,
        vad_filter=True,
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
