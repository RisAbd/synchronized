"""Лёгкий статический сервер для web/ с поддержкой HTTP Range (перемотка аудио).

  python3 server.py            # http://0.0.0.0:8000, корень — web/
  python3 server.py 8080

Дальше, чтобы дать доступ другу:
  ngrok http 8000
и кидаешь выданную https-ссылку.

Только stdlib. Range нужен, чтобы аудио можно было перематывать и стримить.
"""
from __future__ import annotations

import os
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler


class RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + поддержка одиночного Range: bytes=start-end."""

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().do_GET()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().do_GET()

        try:
            unit, _, rng_val = rng.partition("=")
            if unit.strip() != "bytes":
                raise ValueError
            start_s, _, end_s = rng_val.strip().partition("-")
            size = os.path.getsize(path)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                raise ValueError
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{os.path.getsize(path)}")
            self.end_headers()
            return

        length = end - start + 1
        ctype = self.guess_type(path)
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
    handler = partial(RangeHandler, directory=web_dir)
    httpd = HTTPServer(("0.0.0.0", port), handler)
    print(f"synchronized: http://localhost:{port}  (корень: {web_dir})")
    print(f"для доступа другу:  ngrok http {port}")
    print("Ctrl+C — остановить")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлен")


if __name__ == "__main__":
    main()
