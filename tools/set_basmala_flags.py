#!/usr/bin/env python3
"""Выставить per-edition флаги «нужна ли ОТДЕЛЬНАЯ (дорисованная) басмала ﷽» — П9.

Контекст (важно, чтобы будущий я не переоткрывал спор с владельцем):
  • ТЕКСТ РЕДАКЦИЙ ТРОГАТЬ НЕЛЬЗЯ. Каждая колонка (`text` = Tanzil, `text_diyanet` = Diyanet)
    остаётся байт-в-байт как в своём издании. Tanzil ВШИВАЕТ басмалу в текст аята 1 у 113 сур
    (всех, кроме Тавбы-9); Diyanet — НЕТ (басмала в тексте только у Фатихи, где она и есть аят 1).
  • quran.db — переиспользуемый ассет: владелец рисует басмалу САМ в другом проекте. Ему нужен
    per-edition флаг «рисовать ли ﷽ отдельно перед сурой», иначе при Tanzil-тексте (басмала уже
    внутри) + флаг=True получится ДВОЙНАЯ басмала.

Семантика флага = «в тексте ЭТОЙ редакции басмалы НЕТ, потребитель должен дорисовать ﷽ сам»:
  • bismillah_pre         (для `text` / Tanzil):  везде FALSE. Басмала уже в тексте (113 сур) или
                          её нет вовсе (Тавба-9) → отдельная ﷽ не нужна нигде.
  • bismillah_pre_diyanet (для `text_diyanet`):    TRUE у 112 сур; FALSE у [1, 9]. У Фатихи басмала
                          — сам аят 1 (в тексте), у Тавбы басмалы нет → дорисовывать не надо.

Тавба (9) НИКОГДА не True (басмалы у неё нет — владелец特о просил не проставить случайно).
Алайнер НЕ затронут: он работает с `text` (Tanzil), где басмала в тексте есть, поэтому forced
покрывает прочитанную басмалу без изменений; Diyanet-показ ложится через edition_word_map.

Идемпотентно. Запуск (хост):  /usr/bin/python3 tools/set_basmala_flags.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "quran.db"


def main() -> int:
    con = sqlite3.connect(DB)
    cols = [r[1] for r in con.execute("PRAGMA table_info(surahs)")]
    if "bismillah_pre_diyanet" not in cols:
        con.execute("ALTER TABLE surahs ADD COLUMN bismillah_pre_diyanet BOOLEAN")
        print("ALTER: добавлена колонка bismillah_pre_diyanet")

    # Tanzil (`text`): басмала вшита в текст везде, где есть, а у Тавбы её нет → доп. ﷽ не нужна.
    con.execute("UPDATE surahs SET bismillah_pre = 0")
    # Diyanet (`text_diyanet`): доп. ﷽ нужна у всех, КРОМЕ Фатихи(1, басмала=аят 1) и Тавбы(9, нет).
    con.execute("UPDATE surahs SET bismillah_pre_diyanet = (number NOT IN (1, 9))")
    con.commit()

    print("\n=== bismillah_pre (Tanzil `text`) ===")
    for r in con.execute("SELECT bismillah_pre, COUNT(*) c FROM surahs GROUP BY bismillah_pre"):
        print(f"  {r[0]}: {r[1]}")
    print("=== bismillah_pre_diyanet (`text_diyanet`) ===")
    for r in con.execute("SELECT bismillah_pre_diyanet, COUNT(*) c "
                         "FROM surahs GROUP BY bismillah_pre_diyanet"):
        print(f"  {r[0]}: {r[1]}")
    no = [r[0] for r in con.execute("SELECT number FROM surahs WHERE NOT bismillah_pre_diyanet")]
    print("  diyanet False у сур:", sorted(no), "(ожидаем [1, 9])")
    # страховка: Тавба не должна быть True ни в одной редакции
    bad = con.execute("SELECT number FROM surahs WHERE number=9 AND "
                      "(bismillah_pre OR bismillah_pre_diyanet)").fetchall()
    assert not bad, f"Тавба (9) помечена True — ошибка! {bad}"
    print("\nТавба (9): доп. басмала=False в обеих редакциях ✓")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
