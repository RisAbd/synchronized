#!/usr/bin/env python3
"""Построить хранимую таблицу маппинга слов Tanzil↔Diyanet для ВСЕГО Корана (П9).

Владелец хочет quran.db как чистый переиспользуемый ассет с чётким разделением. Здесь мы
превращаем маппинг из «расчёта на лету» в НЕЗАВИСИМУЮ таблицу `edition_word_map`:
    (surah, ayah, tanzil_wi, diyanet_wi)
— по строке на каждую пару сопоставленных слов; diyanet_wi = NULL, если слово Tanzil ни с чем
не сматчено (басмала, приклеенная к аяту 1; отдельные токены-вакфы). Источник карты —
`quran.map_editions` (согласный скелет). Идемпотентно (пересоздаёт таблицу).

Плюс ОТЧЁТ ПО КАЧЕСТВУ по всему Корану (не юнит-тесты, а сверка 6236 аятов):
  • сколько аятов идеально 1:1;
  • сколько со слияниями/дроблениями;
  • сколько слов Tanzil не сматчено (ожидаемо: басмала/вакфы);
  • ⚠ аяты, где остались НЕПОКРЫТЫЕ слова Diyanet (ни один tanzil-wi на них не указывает) —
    это признак рассинхрона; печатаем список, чтобы владелец/я глазами проверил.

Запуск (хост): /usr/bin/python3 tools/build_edition_map.py  [--dry]
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from quran import map_editions, skeleton  # noqa: E402

DB = Path(__file__).resolve().parent.parent / "data" / "quran.db"


def main(dry: bool) -> int:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "select surah_id, number, text, text_diyanet from surah_verses "
        "order by surah_id, number").fetchall()

    if not dry:
        con.execute("DROP TABLE IF EXISTS edition_word_map")
        con.execute("CREATE TABLE edition_word_map ("
                    "surah INTEGER NOT NULL, ayah INTEGER NOT NULL, "
                    "tanzil_wi INTEGER NOT NULL, diyanet_wi INTEGER)")

    n_ayat = perfect = with_group = 0
    unmapped_tanzil = 0
    uncovered_ayat = []          # аяты с непокрытыми словами Diyanet (⚠ рассинхрон)
    total_rows = []
    for r in rows:
        if not r["text_diyanet"]:
            continue
        s, a = r["surah_id"], r["number"]
        tw, dw = r["text"].split(), r["text_diyanet"].split()
        m = map_editions(tw, dw)
        n_ayat += 1
        # запись строк
        for wi, dlist in enumerate(m):
            if dlist:
                for dj in dlist:
                    total_rows.append((s, a, wi, dj))
            else:
                total_rows.append((s, a, wi, None))
                if skeleton(tw[wi]):        # непустое слово Tanzil осталось без пары
                    unmapped_tanzil += 1
        # классификация
        flat = [dj for dlist in m for dj in dlist]
        is_1to1 = (len(tw) == len(dw) and all(len(x) == 1 for x in m)
                   and flat == list(range(len(dw))))
        if is_1to1:
            perfect += 1
        else:
            with_group += 1
        # покрытие слов Diyanet: какие dj не упомянуты никем
        covered = set(flat)
        uncov = [dj for dj in range(len(dw)) if dj not in covered]
        if uncov:
            uncovered_ayat.append((s, a, len(uncov), len(tw), len(dw)))

    if not dry:
        con.executemany("INSERT INTO edition_word_map VALUES (?,?,?,?)", total_rows)
        con.commit()
    con.close()

    print(f"аятов с Diyanet: {n_ayat}")
    print(f"  идеально 1:1: {perfect} ({100*perfect//max(1,n_ayat)}%)")
    print(f"  со слияниями/дроблениями/пропусками: {with_group}")
    print(f"  слов Tanzil без пары (басмала/вакфы — ожидаемо): {unmapped_tanzil}")
    print(f"  строк карты: {len(total_rows)}")
    print(f"\n⚠ аятов с НЕПОКРЫТЫМИ словами Diyanet (потенц. рассинхрон): {len(uncovered_ayat)}")
    for s, a, u, nt, nd in uncovered_ayat[:40]:
        print(f"    {s}:{a}  непокрыто {u}  (Tanzil {nt}, Diyanet {nd})")
    if len(uncovered_ayat) > 40:
        print(f"    … ещё {len(uncovered_ayat)-40}")
    return 0


if __name__ == "__main__":
    sys.exit(main(dry="--dry" in sys.argv))
