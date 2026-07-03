"""Готовит запись для веб-фронта: sync-map + аудио -> web/data/<id>.json + web/audio/ + manifest.

Пример:
  python3 build_web.py --id aswailis-isra \\
      --sync-map work/audio.sync-map.json --audio work/audio.mp3 \\
      --title "Сура Аль-Исра" --title-ar "سورة الإسراء" --reciter "الشيخ يونس اسويلص"

Затем: python3 server.py  (и ngrok http 8000)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from player import build_data
from quran import Quran

WEB = Path(__file__).resolve().parent.parent / "web"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--sync-map", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--title-ar", default="")
    ap.add_argument("--reciter", default="")
    args = ap.parse_args()

    (WEB / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "audio").mkdir(parents=True, exist_ok=True)

    sync_map = json.loads(Path(args.sync_map).read_text())
    q = Quran.load()

    audio_ext = Path(args.audio).suffix
    audio_name = f"{args.id}{audio_ext}"
    shutil.copyfile(args.audio, WEB / "audio" / audio_name)

    data = build_data(sync_map, q, audio_name)
    surah_titles = " · ".join(f"سورة {s['title']}" for s in data["sections"])
    duration = data["timeline"][-1]["t"] if data["timeline"] else 0
    data.update({"id": args.id, "title": args.title or args.id,
                 "title_ar": args.title_ar, "reciter": args.reciter,
                 "duration": round(duration)})
    (WEB / "data" / f"{args.id}.json").write_text(json.dumps(data, ensure_ascii=False))

    # manifest
    man_path = WEB / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.is_file() else {"recitations": []}
    entry = {"id": args.id, "title": args.title or args.id, "title_ar": args.title_ar,
             "reciter": args.reciter, "surahs": surah_titles,
             "audio": audio_name, "duration": round(duration)}
    man["recitations"] = [r for r in man["recitations"] if r["id"] != args.id] + [entry]
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2))

    print(f"готово: web/data/{args.id}.json, web/audio/{audio_name}, manifest обновлён")
    print(f"записей в манифесте: {len(man['recitations'])}")


if __name__ == "__main__":
    main()
