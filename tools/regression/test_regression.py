"""Регресс-тест find_segments БЕЗ GPU: гоняет по сохранённым СКЕЛЕТ-декодам (dec_<id>.txt рядом)
и сверяет с закреплёнными ожиданиями (pins.json). Запускать на ЛЮБОЕ изменение match_align/модели —
ловит «чиню одно, ломаю другое». Фикстуры лежат рядом со скриптом и трекаются в git.

  docker compose exec -T worker python /app/tools/regression/test_regression.py

Пересобрать пины (ТОЛЬКО после осознанной смены поведения, с GPU):
  docker compose exec -T worker python /app/tools/regression/build_regression.py"""
import os, sys, json
sys.path.insert(0, "/app/src"); sys.path.insert(0, "/app/service")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "synchronized.settings")
import django; django.setup()
import match_align
from quran import Quran

REG = os.path.dirname(os.path.abspath(__file__))
quran = Quran.load()
index = match_align.build_index(quran)

def runs_str(segs):
    if not segs: return "None"
    runs = []
    for s, a in segs:
        if runs and runs[-1][-1][0] == s and a == runs[-1][-1][1] + 1:
            runs[-1].append((s, a))
        else:
            runs.append([(s, a)])
    return " | ".join(f"{r[0][0]}:{r[0][1]}-{r[-1][1]}" for r in runs)

pins = json.load(open(f"{REG}/pins.json"))
fails = 0
for rid, expect in sorted(pins.items(), key=lambda kv: int(kv[0])):
    dec = open(f"{REG}/dec_{rid}.txt").read()
    got = runs_str(match_align.find_segments(None, quran, {}, {}, index=index, dec=dec))
    ok = got == expect
    fails += not ok
    print(f"{'OK  ' if ok else 'FAIL'} rec{rid}: ожид={expect!r} получено={got!r}")
print(f"\n{'✅ ВСЕ ПРОШЛИ' if not fails else f'❌ ПРОВАЛОВ: {fails}'} ({len(pins)} рек)")
sys.exit(1 if fails else 0)
