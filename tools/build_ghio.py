"""Собрать статичную выгрузку для GitHub Pages (ветка github.io → сабмодуль syncronized
в risabd.github.io). ТОЛЬКО JSON — никакого аудио: источник каждой записи — YouTube
(youtube_id внутри data.json), плеер встраивает видео.

Гоняется ВНУТРИ docker-воркера через `manage.py shell` (веб на 8000 поднимать не нужно —
порт занят чужим проектом). Данные берём теми же вьюхами, что и живой бэк (RequestFactory →
паритет: та же схема data.json / recitations.json, без дублирования логики). Пишем в
`/app/work/ghio-export` (= хостовое `./work/ghio-export`, `work/` примонтирован).

Запуск (одной командой, ВРУЧНУЮ — не автоматизируем):
    docker compose exec -T worker python manage.py shell < tools/build_ghio.py

Итог `./work/ghio-export/`:
    recitations.json          — список (только ready + с youtube_id, без manual-прогонов)
    r/<id>/data.json          — детализация (forced по умолчанию, audio="", без manual)
    index.html, player.html   — статика (относительные пути ./ — работает из подпапки Pages)

Дальше — вручную (разово): скопировать содержимое в worktree ветки github.io, закоммитить,
запушить; в ../risabd.github.io обновить сабмодуль syncronized. См. docs/DEPLOY.md §6.
"""
import json
import os
import shutil

from django.test import RequestFactory

from recitations import views

OUT = "/app/work/ghio-export"
STATIC = "/app/service/recitations/static"


def _no_manual(runs):
    """Убрать manual-прогоны из списка (владелец: тестовые ручные привязки в выгрузку не тащим)."""
    return [x for x in (runs or []) if x.get("recognizer") != "manual"]


def main():
    rf = RequestFactory()
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    lst = json.loads(views.api_recitations(rf.get("/api/recitations")).content)
    kept = []
    for r in lst["recitations"]:
        rid = r["id"]
        yt = (r.get("youtube_id") or "").strip()
        if r.get("status") != "ready" or not yt:
            print(f"  rec{rid}: status={r.get('status')} youtube={yt!r} — пропуск")
            continue
        r["runs"] = _no_manual(r.get("runs"))
        # forced по умолчанию (prefer=forced) — не даём плееру врубить manual при входе
        data = json.loads(views.data_json(rf.get("/", {"asr": "forced"}), rid).content)
        data["runs"] = _no_manual(data.get("runs"))
        if data.get("active_key") == "manual":  # страховка, если forced вдруг не ready
            data["active_key"] = data["recognizer"] = "forced"
        data["audio"] = ""  # источник только YouTube, mp3 не выгружаем
        d = os.path.join(OUT, "r", str(rid))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "data.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        kept.append(r)
        print(f"  rec{rid}: data.json (active={data.get('active_key')}, YouTube {yt})")

    lst["recitations"] = kept
    with open(os.path.join(OUT, "recitations.json"), "w", encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False)
    print(f"recitations.json: {len(kept)} записей")

    for name in ("index.html", "player.html"):
        shutil.copy(os.path.join(STATIC, name), os.path.join(OUT, name))
    print("статика: index.html, player.html\nготово →", OUT)


main()
