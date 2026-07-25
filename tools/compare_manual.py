"""Сверка прогона-выравнивателя с РУЧНЫМ ЭТАЛОНОМ владельца (прогон 'manual') по ВОЗВРАТАМ и ПОРЯДКУ.

Владелец (25.07, tg_3610): «сверяешь алайнер с моим ручным вводом — он на сколько-то % должен
совпадать ПО ВОЗВРАТАМ И СИНХРОННОСТИ; миллисекунды туда-сюда НЕ учитываются». Этот инструмент:
  1. строит визит-последовательность слов (по t) каждого прогона — [(surah,ayah,wi), …], повторы = дубли;
  2. общее совпадение ПОРЯДКА через difflib.SequenceMatcher.ratio() (игнорит времена, только структуру);
  3. ВОЗВРАТЫ = позиции, где reading-index идёт НАЗАД (rank[i] < rank[i-1]); печатает их в обоих
     прогонах и матчит эталон↔алайнер по слову-приземления (±1 слово) → precision/recall возвратов.

Запуск (tools/ не монтируется в воркер → через stdin + env):
  docker compose exec -T -e REC=7 -e AKEY=w2v worker python - < tools/compare_manual.py
Требует прогон 'manual' у записи (сделай ручным элайнером и «💾 Сохранить в бэк»).
"""
import os
import sys
import difflib

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/service")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "synchronized.settings")
import django  # noqa: E402
django.setup()
from recitations.models import AsrRun  # noqa: E402


def visit_seq(run):
    """Последовательность (surah,ayah,wi) по возрастанию t (повторы — дубли)."""
    wt = sorted(run.data.get("word_timeline", []), key=lambda e: e["t"])
    return [(e["surah"], e["ayah"], e["wi"]) for e in wt]


def rank_map(*seqs):
    """Канонический reading-rank для всех встречающихся слов: сортировка (surah,ayah,wi)."""
    uniq = sorted({t for s in seqs for t in s})
    return {t: i for i, t in enumerate(uniq)}


def returns(seq, rank):
    """Индексы i, где слово идёт НАЗАД по чтению (возврат/перечитка) + слово-приземления."""
    out = []
    for i in range(1, len(seq)):
        if rank[seq[i]] < rank[seq[i - 1]]:
            out.append((i, seq[i]))   # (позиция в последовательности, слово куда вернулись)
    return out


def fmt(t):
    return f"{t[0]}:{t[1]}:{t[2]}"


def main():
    # argv, иначе env REC/AKEY (запуск через stdin `python -` не имеет argv)
    rec_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REC")
    if not rec_arg:
        print("usage: REC=<rec_id> [AKEY=w2v] python - < tools/compare_manual.py", file=sys.stderr)
        return 2
    rec_id = int(rec_arg)
    key = (sys.argv[2] if len(sys.argv) > 2 else os.environ.get("AKEY")) or "w2v"

    man = AsrRun.objects.filter(recitation__pk=rec_id, recognizer="manual").first()
    alg = AsrRun.objects.filter(recitation__pk=rec_id, recognizer=key).first()
    if not man or not man.data:
        print(f"rec{rec_id}: нет ручного эталона (прогон 'manual'). Сделай ручным элайнером + «Сохранить в бэк».")
        return 1
    if not alg or not alg.data:
        print(f"rec{rec_id}: нет прогона '{key}'.")
        return 1

    ms, as_ = visit_seq(man), visit_seq(alg)
    rank = rank_map(ms, as_)
    m_ret, a_ret = returns(ms, rank), returns(as_, rank)

    # общее совпадение ПОРЯДКА (по reading-rank, времена игнорим)
    m_ranks = [rank[t] for t in ms]
    a_ranks = [rank[t] for t in as_]
    ratio = difflib.SequenceMatcher(None, m_ranks, a_ranks).ratio()

    # матч возвратов по слову-приземления ±1 rank
    def near(w1, w2):
        return abs(rank[w1] - rank[w2]) <= 1
    matched = 0
    used = [False] * len(a_ret)
    for _, mw in m_ret:
        for j, (_, aw) in enumerate(a_ret):
            if not used[j] and near(mw, aw):
                used[j] = True
                matched += 1
                break
    recall = matched / len(m_ret) if m_ret else (1.0 if not a_ret else 0.0)
    precision = matched / len(a_ret) if a_ret else (1.0 if not m_ret else 0.0)

    print(f"=== rec{rec_id}: эталон(manual) vs {key} ===")
    print(f"слов-визитов: эталон {len(ms)} | {key} {len(as_)}")
    print(f"совпадение ПОРЯДКА (difflib ratio по reading-rank): {ratio:.3f}")
    print(f"\nВОЗВРАТЫ (перечитки):")
    print(f"  эталон:  {len(m_ret)}  →  " + ", ".join(fmt(w) for _, w in m_ret))
    print(f"  {key:7}: {len(a_ret)}  →  " + ", ".join(fmt(w) for _, w in a_ret))
    print(f"  matched {matched} | recall {recall:.2f} (сколько эталонных возвратов поймал {key}) | "
          f"precision {precision:.2f} (сколько {key}-возвратов реальны)")
    # непойманные эталонные возвраты — что тюнить
    unmatched = [mw for _, mw in m_ret if not any(near(mw, aw) for _, aw in a_ret)]
    if unmatched:
        print(f"\n  ⚠ ПРОПУЩЕНЫ {key}-детектором (тюнинг сюда): " + ", ".join(fmt(w) for w in unmatched))
    return 0


if __name__ == "__main__":
    sys.exit(main())
