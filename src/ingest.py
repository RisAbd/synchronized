"""M2 — получение аудио. URL YouTube -> скачать (yt-dlp); файл -> как есть.

CLI:  python3 ingest.py <url|file> [out_dir]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def fetch(source: str, out_dir: str | Path = "work") -> Path:
    """Возвращает путь к локальному аудиофайлу для source (URL или путь)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not is_url(source):
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(source)
        return p

    # YouTube / прочее — через yt-dlp, только аудио, без плейлиста
    template = str(out_dir / "%(id)s.%(ext)s")
    cmd = ["yt-dlp", "-f", "bestaudio", "--no-playlist",
           "--print", "after_move:filepath", "-o", template, source]
    print(f"[ingest] скачиваю: {source}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp упал:\n{proc.stderr[-800:]}")
    path = proc.stdout.strip().splitlines()[-1]
    return Path(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 ingest.py <url|file> [out_dir]", file=sys.stderr)
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "work"
    print(fetch(sys.argv[1], out))
