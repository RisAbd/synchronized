"""Собрать/пересобрать регресс-фикстуры (ТРЕБУЕТ GPU): для каждой реки сохранить СКЕЛЕТ-ДЕКОД
обычной w2v-модели в dec_<id>.txt (рядом со скриптом) и записать текущий find_segments в pins.json
как ПИН-ожидание. Один GPU-проход. Пересобирать пины ТОЛЬКО когда смена поведения осознанная.
Юзаж: docker compose exec -T worker python /app/tools/regression/build_regression.py [id ...]
       (по умолчанию все реки из media/rec/, у которых есть аудио)"""
import sys, os, json, glob, re
sys.path.insert(0, "/app/src"); sys.path.insert(0, "/app/service")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "synchronized.settings")
import django; django.setup()
import w2v_align, match_align
from quran import Quran

REG = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1:]:
    IDS = [int(x) for x in sys.argv[1:]]
else:
    IDS = sorted(int(m.group(1)) for d in glob.glob("/app/media/rec/*")
                 if (m := re.search(r"/(\d+)$", d)))
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

pins = {}
for rid in IDS:
    ap = None
    for ext in ("mp3", "ogg", "wav", "m4a", "opus"):
        p = f"/app/media/rec/{rid}/audio.{ext}"
        if os.path.exists(p): ap = p; break
    if not ap:
        print(f"rec{rid}: НЕТ аудио — пропуск"); continue
    E, stride, idx2ch, ch2idx = w2v_align.emissions(ap)
    special = {ch2idx.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>", "|", "-", "ـ")} - {None}
    dec = match_align.greedy_skeleton(E, idx2ch, special)
    open(f"{REG}/dec_{rid}.txt", "w").write(dec)
    rs = runs_str(match_align.find_segments(None, quran, {}, {}, index=index, dec=dec))
    pins[str(rid)] = rs
    print(f"rec{rid}: declen={len(dec)} → {rs}")
    del E
json.dump(pins, open(f"{REG}/pins.json", "w"), ensure_ascii=False, indent=2)
print("PINS saved:", pins)
