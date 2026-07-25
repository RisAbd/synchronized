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


def _fetch(rf, rid, key=None):
    """data.json прогона через живую вьюху (RequestFactory — паритет с бэком). key=None → дефолт."""
    q = {"asr": key} if key else {}
    data = json.loads(views.data_json(rf.get("/", q), rid).content)
    data["audio"] = ""  # источник только YouTube, mp3 не выгружаем
    return data


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
        d = os.path.join(OUT, "r", str(rid))
        os.makedirs(d, exist_ok=True)
        # data.json = ДЕФОЛТНЫЙ прогон (active_run сам выбирает по приоритету/мин.прыжкам)
        data = _fetch(rf, rid)
        with open(os.path.join(d, "data.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        # ПОФАЙЛОВО каждый ready-прогон (вкл. manual/test/test2) → r/<id>/<key>.json;
        # на статике так переключается любой прогон (?asr= там игнорируется).
        run_keys = [x.get("recognizer") for x in (r.get("runs") or [])
                    if x.get("status") == "ready" and x.get("recognizer")]
        for key in run_keys:
            rd = _fetch(rf, rid, key)
            with open(os.path.join(d, f"{key}.json"), "w", encoding="utf-8") as f:
                json.dump(rd, f, ensure_ascii=False)
        kept.append(r)
        print(f"  rec{rid}: data.json (active={data.get('active_key')}) + прогоны {run_keys}, YouTube {yt}")

    lst["recitations"] = kept
    with open(os.path.join(OUT, "recitations.json"), "w", encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False)
    print(f"recitations.json: {len(kept)} записей")

    for name in ("index.html", "player.html"):
        shutil.copy(os.path.join(STATIC, name), os.path.join(OUT, name))
    print("статика: index.html, player.html\nготово →", OUT)


main()
