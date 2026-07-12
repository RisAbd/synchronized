#!/usr/bin/env python3
"""Вернуть ре-импортированным рекам (recitation) ПРЕЖНИЕ id, чтобы вся история
замечаний (rec7 خضرا, rec11 Ан-Наджм 0:50 и т.п.) снова билась.

Маппинг current_pk -> old_pk (по youtube_id, порядок ре-импорта):
  1(lD40MsqrI6I)->5  2(TcX_XDiMJ8E)->6  3(KWKdIF8_ziU)->7
  4(oM1e8fILtaM)->9  5(rsPWlngoL3s)->10  6(hMfYDkmIZ3Y)->11

Запускать ТОЛЬКО когда прогоны завершены (не на лету — celery-задачи держат rec_id).
  docker compose exec -T worker python manage.py shell < tools/remap_ids.py
"""
from django.db import connection

MAP = {1: 5, 2: 6, 3: 7, 4: 9, 5: 10, 6: 11}
OFFSET = 1000

cur = connection.cursor()
cur.execute("PRAGMA foreign_keys=OFF")
with connection.constraint_checks_disabled():
    # 1) сдвинуть всё во временный диапазон (без коллизий)
    cur.execute("UPDATE recitations_recitation SET id = id + %s" % OFFSET)
    cur.execute("UPDATE recitations_asrrun SET recitation_id = recitation_id + %s" % OFFSET)
    # 2) временные -> финальные
    for old, new in MAP.items():
        cur.execute("UPDATE recitations_recitation SET id=%s WHERE id=%s", [new, old + OFFSET])
        cur.execute("UPDATE recitations_asrrun SET recitation_id=%s WHERE recitation_id=%s", [new, old + OFFSET])
cur.execute("PRAGMA foreign_keys=ON")

from recitations.models import Recitation
print("после ремапа:", [(r.id, r.youtube_id) for r in Recitation.objects.order_by("id")])
