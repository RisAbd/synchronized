"""Аудит инварианта чтеца по всем прогонам — объективно, без звука и без GPU.

Правило владельца: подсветка идёт вперёд ПО ОДНОМУ слову либо возвращается НАЗАД (перечитка);
резкого прыжка ВПЕРЁД через слова физически не бывает. `pipeline.alignment_invariants` считает это
по финальному word_timeline (плоский индекс слова: соседи по времени = +1 или ≤0, но не ≥+2).

Печатает по каждому прогону число прыжков вперёд + схлопнутых слов и первые случаи с mm:ss.
С `--write` дописывает счётчики в run.metrics (backfill старых прогонов без перегона на GPU).

Запуск в docker-воркере:
    docker compose exec -T worker python manage.py shell < tools/audit_alignment.py
    docker compose exec -T worker python manage.py shell -c "import tools.audit_alignment" ...
(или как модуль; ниже — самодостаточный скрипт под `manage.py shell < …`)."""
import os

WRITE = os.environ.get("SYNC_AUDIT_WRITE") == "1"   # manage.py shell не пробрасывает argv → env

from recitations.models import Recitation      # noqa: E402
from recitations import pipeline                 # noqa: E402


def mmss(t):
    t = float(t or 0)
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main():
    total_fwd = 0
    for rec in Recitation.objects.order_by("id"):
        for run in rec.runs.all():
            if run.status != "ready" or not run.data:
                continue
            inv = pipeline.alignment_invariants(run.data)
            fwd, col = inv["forward_jumps"], inv["collapsed_words"]
            total_fwd += fwd
            flag = "  ⚠️" if fwd else ""
            print(f"rec{rec.id:<3} {run.recognizer:<8} прыжков_вперёд={fwd:<3} схлопнутых={col:<3}{flag}")
            for j in inv["forward_jumps_detail"][:8]:
                print(f"       {mmss(j['t'])}  {j['from']} → {j['to']}  (пропущено {j['skip']})")
            if WRITE:
                m = dict(run.metrics or {})
                m["forward_jumps"] = fwd
                m["collapsed_words"] = col
                m["invariants"] = inv
                run.metrics = m
                run.save(update_fields=["metrics", "updated_at"])
    print(f"\nИТОГО прыжков вперёд по всем прогонам: {total_fwd}"
          + ("  (записано в metrics)" if WRITE else "  (--write чтобы записать)"))


main()
