#!/usr/bin/env python3
"""Импорт текста мусхафа Diyanet (турецкая редакция) в data/quran.db как ВТОРАЯ редакция.

Зачем: владелец хочет вид турецкого мусхафа. Текст Diyanet отличается от нашего Tanzil в слое
dabt (огласовки/вакфы: U+06EA вместо кясры, дагер-алиф, пометка нач. хамзы, знаки остановок) —
подтверждено сверкой (см. docs/RESEARCH-turkish-mushaf.md). Лицензия источника CC BY-NC-SA
(некоммерч.) — владелец подтвердил, что проект некоммерческий («это ж Куран»).

Источник: api.acikkuran.com (Diyanet-based). Кладём в НОВУЮ колонку surah_verses.text_diyanet,
матчим по (surah_id, number). Наш Tanzil-текст (колонка text) НЕ трогаем. Слова/дробление у
редакций местами разное — маппинг (Задача A) считается отдельно, здесь только сырой текст.

Запуск (хост, есть сеть): /usr/bin/python3 tools/import_diyanet.py
Идемпотентно: колонку создаёт если нет, перезаписывает значения. Сверяет число аятов по суре,
НЕ пишет суру при рассинхроне (сообщает). Опция --dry — только проверить, без записи.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "quran.db"
API = "https://api.acikkuran.com/surah/{}"


def fetch_surah(n: int) -> list[dict]:
    req = urllib.request.Request(API.format(n), headers={"User-Agent": "synchronized/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode())
    return d["data"]["verses"]


def main(dry: bool) -> int:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    # сколько аятов в каждой суре по НАШЕЙ базе (эталон для сверки)
    ours = {}
    for r in con.execute("select surah_id, count(*) c from surah_verses group by surah_id"):
        ours[r["surah_id"]] = r["c"]
    n_surahs = len(ours)
    print(f"наших сур: {n_surahs}, аятов: {sum(ours.values())}")

    if not dry:
        cols = [c[1] for c in con.execute("PRAGMA table_info(surah_verses)")]
        if "text_diyanet" not in cols:
            con.execute("ALTER TABLE surah_verses ADD COLUMN text_diyanet TEXT")
            print("+ колонка text_diyanet")

    mism, written, missing = [], 0, 0
    for s in range(1, n_surahs + 1):
        try:
            verses = fetch_surah(s)
        except Exception as e:
            print(f"  сура {s}: ошибка загрузки {e}")
            mism.append(s)
            continue
        if len(verses) != ours.get(s):
            print(f"  ⚠ сура {s}: аятов у Diyanet {len(verses)} ≠ наших {ours.get(s)} — ПРОПУСК")
            mism.append(s)
            continue
        if not dry:
            for v in verses:
                cur = con.execute(
                    "update surah_verses set text_diyanet=? where surah_id=? and number=?",
                    (v["verse"], s, v["verse_number"]))
                if cur.rowcount == 1:
                    written += 1
                else:
                    missing += 1
        time.sleep(0.15)   # вежливо к API
        if s % 20 == 0:
            print(f"  ... {s}/{n_surahs}")

    if not dry:
        con.commit()
    # проверка покрытия
    filled = con.execute("select count(*) from surah_verses where text_diyanet is not null "
                         "and text_diyanet<>''").fetchone()[0]
    total = con.execute("select count(*) from surah_verses").fetchone()[0]
    con.close()
    print(f"\nзаписано: {written}, не сматчено: {missing}, рассинхрон-сур: {len(mism)} {mism or ''}")
    print(f"покрытие text_diyanet: {filled}/{total}")
    return 0 if (dry or (filled == total and not mism)) else 1


if __name__ == "__main__":
    sys.exit(main(dry="--dry" in sys.argv))
