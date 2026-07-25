"""Идея владельца tg_4343/4364: он даёт РАСШИРЕННЫЙ ТЕКСТ чтения (канон + продублированные повторы,
сплошняком) → маппим на канонические позиции (surah:ayah:wi) с пометкой rep у повторных вхождений →
forced-Viterbi (slots) по акустике. Быстрее ручной подгонки, структуру задаёт он.

Маппер: канон диапазона = последовательность слов; идём по тексту владельца, каждый токен матчим по
СКЕЛЕТУ (согласные) вперёд (окно +2, допускает пропуск) либо назад (окно 12 — это ПОВТОР, rep=True).

Использование:
  echo "<арабский расширенный текст>" | docker compose exec -T -e REC=7 worker python /app/work/forced_from_text.py
или для само-проверки: docker compose exec -T -e REC=7 -e SELFTEST=1 worker python /app/work/forced_from_text.py
"""
import os, sys
sys.path.insert(0, "/app/src"); sys.path.insert(0, "/app/service")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "synchronized.settings")
import django; django.setup()
import numpy as np, w2v_align
from quran import Quran, word_tokens, skeleton
from recitations.models import AsrRun
from recitations import pipeline

REC = int(os.environ.get("REC", "7"))
q = Quran.load()

# диапазон аятов: берём из авто-w2v прогона (он верен), можно переопределить env RANGE="101:1-103:5"
run_w2v = AsrRun.objects.get(recitation__pk=REC, recognizer="w2v")
wt_auto = sorted(run_w2v.data["word_timeline"], key=lambda e: e["t"])
rng_pairs = sorted({(e["surah"], e["ayah"]) for e in wt_auto})
canon = []   # (s,a,wi,word,skel)
for s, a in rng_pairs:
    for wi, w in enumerate(word_tokens(q.surah(s).verses[a-1].text)):
        canon.append((s, a, wi, w, skeleton(w)))
print(f"канон диапазона: {len(canon)} слов ({rng_pairs[0]}..{rng_pairs[-1]})", file=sys.stderr)

def map_text_to_slots(tokens):
    """tokens → slots [(s,a,wi,word,rep)]. Слово может встречаться в диапазоне НЕСКОЛЬКО раз
    (وهو/الأبصار/شيء…) → среди всех канон-позиций того же скелета выбираем по:
      (1) LOOKAHEAD: позиция, с которой СЛЕДУЮЩЕЕ слово продолжается вперёд (canon[q+1]==next),
          — это держит фразу-перечитку цельной (напр. второе «وهو يدرك الأبصار» → 3,4,5, а не 6,4,5);
      (2) при равенстве — ближе к ожидаемой позиции p (= прошлая+1), т.е. меньше прыжок.
    rep=True если КЛЮЧ (s,a,wi) уже встречался (конвенция build_data — любое повторное вхождение)."""
    sks = [skeleton(t) for t in tokens]
    slots = []; p = 0; seen = set(); unmatched = 0
    for i, sk in enumerate(sks):
        cands = [q for q in range(len(canon)) if canon[q][4] and canon[q][4] == sk]
        if not cands:
            unmatched += 1; continue
        nxt = sks[i+1] if i+1 < len(sks) else None
        def score(q):
            cont = 1 if (nxt and q+1 < len(canon) and canon[q+1][4] == nxt) else 0
            return (cont, -abs(q - p))           # сперва продолжаемость фразы, потом близость к p
        q = max(cands, key=score)
        s, a, wi, w, _ = canon[q]
        rep = (s, a, wi) in seen; seen.add((s, a, wi))
        slots.append((s, a, wi, w, rep)); p = q + 1
    return slots, unmatched

if os.environ.get("SELFTEST"):
    # само-проверка: «текст» = слова из ручного прогона по порядку → маппер должен воспроизвести слоты
    run_m = AsrRun.objects.get(recitation__pk=REC, recognizer="manual")
    seq = sorted(run_m.data["word_timeline"], key=lambda e: e["t"])
    text_tokens = []
    for e in seq:
        toks = word_tokens(q.surah(e["surah"]).verses[e["ayah"]-1].text)
        if 0 <= e["wi"] < len(toks): text_tokens.append(toks[e["wi"]])
    slots, un = map_text_to_slots(text_tokens)
    reps = sum(1 for x in slots if x[4])
    print(f"SELFTEST: вход {len(text_tokens)} слов → слотов {len(slots)}, reps {reps}, не сматчено {un}")
    print(f"  (ручной эталон: {len(seq)} слов, reps {sum(1 for e in seq if e.get('rep'))})")
    sys.exit(0)

raw = sys.stdin.read().strip()
if not raw:
    print("нет текста на stdin", file=sys.stderr); sys.exit(1)
tokens = raw.split()
slots, un = map_text_to_slots(tokens)
print(f"вход {len(tokens)} слов → слотов {len(slots)}, reps {sum(1 for x in slots if x[4])}, не сматчено {un}", file=sys.stderr)

E = np.load(f"/app/work/rec{REC}_w2v_emis.npy").astype("float32")
dur = (run_w2v.data or {}).get("duration") or (E.shape[0]*0.02)
stride = dur/E.shape[0]*1000.0
from transformers import Wav2Vec2Processor
proc = Wav2Vec2Processor.from_pretrained(w2v_align._MODEL_NAME)
vocab = proc.tokenizer.get_vocab()
idx2ch = {int(v):k for k,v in vocab.items()}; ch2idx = {k:int(v) for k,v in vocab.items()}
verses = [(s,a,q.surah(s).verses[a-1].text) for s,a in rng_pairs]
res = w2v_align.forced_align(E, stride, verses, idx2ch, ch2idx, f"/app/media/rec/{REC}/audio.mp3", slots=slots)
m = res["meta"]; wt2 = res["word_timeline"]
fj = pipeline.alignment_invariants({"word_timeline": wt2}).get("forward_jumps")
print(f"forced-по-тексту: wt={len(wt2)} cov={m.get('coverage')} fj={fj} reps={m.get('reps')}")

# ключевые места для глаза
for a,wis,nm in [(101,(0,1,2),"بديع(перечитка)"),(99,(32,),"انظروا×3"),(99,(22,),"قنوان×2")]:
    hits=[e for e in wt2 if e["ayah"]==a and e["wi"] in wis]
    print(f"  {nm}: "+", ".join(f"{e['wi']}@{e['t']:.1f}{'R' if e.get('rep') else ''}" for e in hits))

SAVE = os.environ.get("SAVE")
if SAVE:
    run_obj, _ = AsrRun.objects.get_or_create(recitation=run_w2v.recitation, recognizer=SAVE)
    pipeline.build_manual_run(run_obj, wt2)   # тот же путь, что ручной: sync_map→build_data→save
    run_obj.refresh_from_db()
    print(f"сохранено под '{SAVE}': status={run_obj.status} wt={len(run_obj.data.get('word_timeline',[]))}")
