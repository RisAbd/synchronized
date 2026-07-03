"""Связка всего конвейера: URL/файл -> (скачать) -> ASR -> align -> плеер.

Примеры:
  python3 run.py https://www.youtube.com/watch?v=oM1e8fILtaM
  python3 run.py path/to/recitation.mp3
  python3 run.py work/audio.mp3 --lang ar

Результат: work/<name>.player.html + work/<name>.sync-map.json + work/<name>.transcript.json.
Открой .player.html в браузере (аудио подключается рядом лежащим файлом).

ВАЖНО: перед запуском выставить LD_LIBRARY_PATH для cuDNN (см. asr.py / skill audio-task),
иначе ASR упадёт. Обёртка ниже пытается выставить его сама, если не задан.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def _ensure_cudnn_path():
    """Подставляем pip-путь к cuDNN/cuBLAS, если LD_LIBRARY_PATH не настроен."""
    if "cudnn" in os.environ.get("LD_LIBRARY_PATH", ""):
        return
    import site
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        cudnn = Path(base) / "nvidia" / "cudnn" / "lib"
        cublas = Path(base) / "nvidia" / "cublas" / "lib"
        if cudnn.is_dir():
            os.environ["LD_LIBRARY_PATH"] = (
                f"{cudnn}:{cublas}:" + os.environ.get("LD_LIBRARY_PATH", "")
            )
            return


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    source = sys.argv[1]
    lang = "ar"
    if "--lang" in sys.argv:
        lang = sys.argv[sys.argv.index("--lang") + 1]

    work = Path(__file__).resolve().parent.parent / "work"
    work.mkdir(exist_ok=True)

    _ensure_cudnn_path()
    import ingest
    import asr
    import align as align_mod
    from player import build as build_player
    from quran import Quran

    # 1. ingest
    audio = ingest.fetch(source, work)
    print(f"[1/4] аудио: {audio}")

    stem = audio.stem
    # аудио должно лежать рядом с плеером (в work/), плеер ссылается относительным путём
    audio_local = work / audio.name
    if audio.resolve() != audio_local.resolve():
        shutil.copyfile(audio, audio_local)

    # 2. asr
    tr_path = work / f"{stem}.transcript.json"
    if tr_path.is_file():
        print(f"[2/4] транскрипт есть, переиспользую: {tr_path}")
        transcript = json.loads(tr_path.read_text())
    else:
        print("[2/4] транскрибирую (faster-whisper, GPU)...")
        transcript = asr.transcribe(str(audio_local), language=lang)
        tr_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    print(f"      слов: {len(transcript['words'])}")

    # 3. align
    print("[3/4] выравниваю на канон...")
    q = Quran.load()
    words = align_mod.load_transcript(tr_path)
    sync_map = align_mod.align(words, q)
    sm_path = work / f"{stem}.sync-map.json"
    sm_path.write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))
    m = sync_map["meta"]
    print(f"      привязано {m['aligned_points']}/{m['asr_words']} "
          f"({m['coverage']*100:.0f}%), сегментов {m['segments']}")

    # 4. player
    print("[4/4] собираю плеер...")
    html = build_player(sync_map, q, audio_local.name)
    player_path = work / f"{stem}.player.html"
    player_path.write_text(html)

    print(f"\nГотово. Открой в браузере:\n  {player_path.resolve()}")


if __name__ == "__main__":
    main()
