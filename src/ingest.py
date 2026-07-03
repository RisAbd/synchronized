"""M2 — получение аудио. URL YouTube -> скачать (yt-dlp); файл -> как есть.

Анти-бот YouTube («content is not available on this app») обходим перебором player_client
(android часто проходит там, где web/tv блокируются). Формат — bestaudio/best (fallback на
комбинированный, если отдельной аудио-дорожки нет), затем извлекаем звук в mp3.

CLI:  python3 ingest.py <url|file> [out_dir]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# порядок перебора клиентов: android первым (обычно пробивает анти-бот на старых yt-dlp)
YT_CLIENTS = ["android", "default", "ios", "tv", "web_safari", "mweb"]


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _ytdlp_bin() -> str:
    return shutil.which("yt-dlp") or os.path.join(os.path.expanduser("~"), ".local/bin/yt-dlp")


def fetch(source: str, out_dir: str | Path = "work") -> Path:
    """Возвращает путь к локальному аудиофайлу для source (URL или путь)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not is_url(source):
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(source)
        return p

    ytdlp = _ytdlp_bin()
    template = str(out_dir / "%(id)s.%(ext)s")
    clients = YT_CLIENTS.copy()
    forced = os.environ.get("SYNC_YT_CLIENT")
    if forced:
        clients = [forced]

    # Попытки: сначала клиенты без кук; затем — тот же перебор с куками из браузера
    # (владелец разрешил переиспользовать firefox-сессию). SYNC_YT_COOKIES_BROWSER
    # форсит браузер; иначе по умолчанию пробуем firefox как последний резерв.
    cookies_browser = os.environ.get("SYNC_YT_COOKIES_BROWSER", "firefox")
    attempts = [(c, None) for c in clients] + [(c, cookies_browser) for c in clients]

    errors = []
    for client, cookies in attempts:
        cmd = [ytdlp, "-f", "bestaudio/best", "--no-playlist", "--no-warnings",
               "-x", "--audio-format", "mp3",
               "--extractor-args", f"youtube:player_client={client}",
               "--print", "after_move:filepath", "-o", template, source]
        if cookies:
            cmd[1:1] = ["--cookies-from-browser", cookies]
        tag = f"{client}+{cookies}" if cookies else client
        print(f"[ingest] yt-dlp (client={tag}): {source}", file=sys.stderr)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return Path(proc.stdout.strip().splitlines()[-1])
        errors.append(f"[{tag}] {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'no output'}")

    raise RuntimeError("yt-dlp не смог скачать (перепробованы клиенты и куки):\n" + "\n".join(errors[-4:]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 ingest.py <url|file> [out_dir]", file=sys.stderr)
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "work"
    print(fetch(sys.argv[1], out))
