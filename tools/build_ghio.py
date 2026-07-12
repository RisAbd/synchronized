#!/usr/bin/env python3
"""Собрать статичную выгрузку для GitHub Pages (ветка github.io → сабмодуль syncronized
в risabd.github.io). ТОЛЬКО JSON — никакого аудио: источник каждой записи — YouTube
(youtube_id внутри data.json), плеер встраивает видео. Записи без YouTube пропускаются.

Тянет с ЖИВОГО бэка (localhost:8000, поднятый docker web):
  • /api/recitations  → recitations.json (только ready + с youtube_id)
  • /r/<id>/data.json → r/<id>/data.json (audio="" — источник только YouTube)
  • index.html, player.html — пропатченная статика (относительные пути ./) в корень.

Использование:  python3 tools/build_ghio.py <out-dir>  (по умолч. localhost:8000).
Плеер грузит data.json ОТНОСИТЕЛЬНО (см. DATA_BASE в player.html), поэтому работает из
любой подпапки GitHub Pages без правок.
"""
import json, os, shutil, sys, urllib.request

BASE = os.environ.get("SYNC_API", "http://localhost:8000")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(REPO, "service", "recitations", "static")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "ghio-export")


def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return r.read(), r.headers


def main():
    os.makedirs(OUT, exist_ok=True)
    lst = json.loads(get("/api/recitations")[0])
    recs = lst.get("recitations", [])

    kept = []
    for r in recs:
        rid = r["id"]
        if r.get("status") != "ready":
            print(f"  rec{rid}: status={r.get('status')} — пропуск"); continue
        data = json.loads(get(f"/r/{rid}/data.json")[0])
        yt = (data.get("youtube_id") or "").strip()
        if not yt:
            print(f"  rec{rid}: нет youtube_id (локальная загрузка) — пропуск"); continue
        d = os.path.join(OUT, "r", str(rid)); os.makedirs(d, exist_ok=True)
        data["audio"] = ""   # источник только YouTube, mp3 не выгружаем
        with open(os.path.join(d, "data.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        kept.append(r)
        print(f"  rec{rid}: data.json (YouTube {yt})")

    lst["recitations"] = kept
    with open(os.path.join(OUT, "recitations.json"), "w", encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False)
    print(f"recitations.json: {len(kept)} записей (из {len(recs)})")

    for name in ("index.html", "player.html"):
        shutil.copy(os.path.join(STATIC, name), os.path.join(OUT, name))
    print("статика: index.html, player.html\nготово →", OUT)


if __name__ == "__main__":
    main()
