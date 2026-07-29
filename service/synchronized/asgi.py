import os
import sys

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "synchronized.settings")
for _p in ("/app/src", "/app/service"):           # ядро (match_align/w2v_align/quran) для WS-пути
    if _p not in sys.path:
        sys.path.insert(0, _p)

_django = get_asgi_application()


async def application(scope, receive, send):
    """ASGI-роутер: websocket /live/ws → live_ws (стриминг WI), всё остальное (http) → Django.
    Тонкий, без Channels — один ws-путь, HTTP-часть Django не трогаем."""
    if scope.get("type") == "websocket" and scope.get("path") == "/live/ws":
        from recitations.live_views import live_ws
        await live_ws(scope, receive, send)
        return
    await _django(scope, receive, send)
