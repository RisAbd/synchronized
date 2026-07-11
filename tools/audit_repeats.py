#!/usr/bin/env python3
"""Аудит возвратов чтеца (П8) по forced-прогонам — объективно, без прослушивания.

Читает `media/rec/<id>/asr/forced/sync-map.json` и печатает КАЖДЫЙ возврат назад
(точки `rep:True`, вставленные `falign._detect_repeats`), сгруппированные в события:
слово-остановка перед разрывом → траектория возврата → возврат «вперёд».

Зачем: тюнинг П8 капризный и регрессит (см. docs/BACKLOG «Известные баги»). Когда
владелец даёт таймкод с новой ссылки — сверяем структуру возврата здесь БЕЗ звука:
дикие прыжки (назад более чем на 1 аят, далеко по словам) сразу видны и помечаются ⚠.

Использование:
    python3 tools/audit_repeats.py            # все записи из ./media
    python3 tools/audit_repeats.py 11         # только rec11
    SYNC_MEDIA_ROOT=/path python3 tools/audit_repeats.py

Только чтение (никаких изменений в БД/файлах).
"""
import json
import os
import sys
from pathlib import Path

MEDIA_ROOT = Path(os.environ.get("SYNC_MEDIA_ROOT", "media"))

# Порог, за которым возврат стоит переслушать (эвристика, не приговор):
_WARN_AYAH_BACK = 2      # прыжок назад более чем на столько аятов
_WARN_WORD_BACK = 12     # прыжок назад более чем на столько слов внутри аята


def mmss(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"


def ref(p: dict) -> str:
    return f"{p['surah']}:{p['ayah']} wi{p['wi']}"


def _ayah_key(p: dict):
    return (p["surah"], p["ayah"])


def audit_one(rec_id: str, path: Path) -> int:
    """Печатает возвраты для одной записи, возвращает число ⚠-флагов."""
    sm = json.loads(path.read_text())
    wt = sorted(sm.get("word_timeline", []), key=lambda w: w["t"])
    meta = sm.get("meta", {})
    reps = [i for i, w in enumerate(wt) if w.get("rep")]
    hdr = (f"=== rec{rec_id} === repeats_inserted(meta)={meta.get('repeats_inserted', '?')} "
           f"respaced={meta.get('stuffed_respaced', '?')} точек_wt={len(wt)}")
    print(hdr)
    if not reps:
        print("  (возвратов нет)\n")
        return 0

    # группируем rep-точки в события: соседние по индексу rep-точки = один возврат
    events, cur = [], [reps[0]]
    for a, b in zip(reps, reps[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            events.append(cur); cur = [b]
    events.append(cur)

    warns = 0
    for ev in events:
        first = wt[ev[0]]
        # слово-остановка = последняя НЕ-rep точка перед событием
        stop = next((wt[j] for j in range(ev[0] - 1, -1, -1) if not wt[j].get("rep")), None)
        # куда возврат «вперёд» после события
        nxt = wt[ev[-1] + 1] if ev[-1] + 1 < len(wt) else None

        traj = " → ".join(ref(wt[j]) for j in ev)
        stop_s = f"{ref(stop)} @{mmss(stop['t'])}" if stop else "?"
        print(f"  ▸ возврат @{mmss(first['t'])}: остановка {stop_s}")
        print(f"      назад: {traj}")
        if nxt:
            print(f"      вперёд: {ref(nxt)} @{mmss(nxt['t'])}")

        # эвристика «дикости»: насколько далеко назад от слова-остановки
        if stop:
            da = (stop["surah"], stop["ayah"])
            ta = (first["surah"], first["ayah"])
            if da[0] == ta[0]:
                ayah_back = da[1] - first["ayah"]
                word_back = (stop["wi"] - first["wi"]) if da == ta else None
                if ayah_back > _WARN_AYAH_BACK:
                    print(f"      ⚠ назад на {ayah_back} аята — переслушать (возможен over-reach)")
                    warns += 1
                elif word_back is not None and word_back > _WARN_WORD_BACK:
                    print(f"      ⚠ назад на {word_back} слов внутри аята — переслушать")
                    warns += 1
    print()
    return warns


def main():
    argv = sys.argv[1:]
    if argv:
        ids = argv
    else:
        ids = sorted((d.name for d in (MEDIA_ROOT / "rec").iterdir() if d.is_dir()),
                     key=lambda x: int(x) if x.isdigit() else x)
    total_warns = 0
    found = 0
    for rid in ids:
        path = MEDIA_ROOT / "rec" / str(rid) / "asr" / "forced" / "sync-map.json"
        if not path.is_file():
            print(f"=== rec{rid} === (нет forced sync-map)\n")
            continue
        found += 1
        total_warns += audit_one(str(rid), path)
    print(f"Итого: {found} записей, ⚠-флагов к прослушиванию: {total_warns}")


if __name__ == "__main__":
    main()
