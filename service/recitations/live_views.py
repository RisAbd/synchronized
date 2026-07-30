"""Live-демо WI (владелец tg_5060: удалённый тест стриминга): аудио с микрофона → место в Коране
В РЕАЛЬНОМ ВРЕМЕНИ. Полный WI-пайплайн на входящем аудио: эмиссии (GPU) → greedy-декод →
`find_segments` (какой пассаж читается) → `SegmentTracker` (позиция внутри). Возвращает найденный
пассаж + текущий аят + арабский текст. Отдельный GPU-сервис `live` на :8010, наружу через ngrok."""
import os
import subprocess
import tempfile

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

_CACHE = {}
_SESS = {}                                                  # sid → последний найденный ответ (антимерцание)
_TAIL_SEC = float(os.environ.get("SYNC_LIVE_TAIL", "22"))   # берём последние N с аудио (свежий контекст)


def _quran():
    if "q" not in _CACHE:
        from quran import Quran
        _CACHE["q"] = Quran.load()
    return _CACHE["q"]


def _index():
    if "idx" not in _CACHE:
        import match_align
        _CACHE["idx"] = match_align.build_index(_quran())
    return _CACHE["idx"]


def _vocab():
    if "vocab" not in _CACHE:
        import w2v_align
        from transformers import Wav2Vec2Processor
        proc = Wav2Vec2Processor.from_pretrained(w2v_align._MODEL_NAME)
        v = proc.tokenizer.get_vocab()
        _CACHE["vocab"] = ({int(x): k for k, x in v.items()}, {k: int(x) for k, x in v.items()})
    return _CACHE["vocab"]


def _ayah_text(q, s, a):
    try:
        return q.surah(s).verses[a - 1].text
    except Exception:
        return ""


def _surah_payload(q, s):
    """ВЕСЬ текст суры s → {surah, title, ayat:[{ayah,text}]}. Фронт рендерит суру ЦЕЛИКОМ один раз
    (стабильный DOM, как плеер), а трек-тики лишь двигают подсветку+скролл — без перерисовки на
    каждый ответ бэка (владелец 30.07: перерисовка дёргала позицию текста на экране, терялось место)."""
    try:
        su = q.surah(s)
        ayat = []
        for v in su.verses:
            item = {"ayah": v.ayah, "text": v.text}
            dv = getattr(v, "text_diyanet", "") or ""   # тур. мусхаф (Diyanet) — если есть в quran.db
            if dv:
                item["text_diyanet"] = dv
            ayat.append(item)
        return {"surah": s, "title": getattr(su, "title", str(s)), "ayat": ayat}
    except Exception:
        return None


@csrf_exempt
def live_locate(request):
    """POST: тело = аудио-блоб (webm/opus/wav). Ответ JSON: найденный пассаж + текущий аят + текст."""
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    sid = request.GET.get("sid", "") or "_"
    raw = request.body
    if not raw or len(raw) < 2000:
        return JsonResponse(_continuity(sid, {"ok": True, "empty": True}))

    try:
        import json as _json
        res = _do_locate(raw)
        payload = _json.loads(res.content)
        return JsonResponse(_continuity(sid, payload))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("LIVE_LOCATE_ERROR:\n" + tb, flush=True)   # видно в логах (монитор)
        return JsonResponse(_continuity(sid, {"ok": False,
                            "error": type(e).__name__ + ": " + str(e), "trace": tb[-900:]}))


_SWITCH_CONFIRM = int(os.environ.get("SYNC_LIVE_CONFIRM", "2"))   # окон подряд для смены суры


def _primary_surah(p):
    if p.get("current"):
        return p["current"]["surah"]
    segs = p.get("segments") or []
    return segs[0]["surah"] if segs else None


def _continuity(sid, cur):
    """Контекст-aware удержание позиции (директива владельца: не «тупо первый кандидат»).
    • не нашли сейчас → держим последний найденный (антимерцание, stale);
    • нашли ТУ ЖЕ суру → принимаем;
    • нашли ДРУГУЮ суру → это либо реальный переход, либо разовый ложный скачок короткого окна →
      принимаем ТОЛЬКО если подтвердилось _SWITCH_CONFIRM окон подряд; иначе держим прошлое."""
    sess = _SESS.setdefault(sid, {"last": None, "pk": None, "pn": 0})
    last = sess["last"]
    if not cur.get("found"):
        if last:
            out = dict(last); out["stale"] = True; return out
        return cur
    cs = _primary_surah(cur)
    if last is None or _primary_surah(last) == cs:
        sess["last"] = cur; sess["pk"] = None; sess["pn"] = 0
        return cur
    # другая сура — требуем подтверждения (континуитет-гейт против ложных скачков короткого окна)
    if sess["pk"] == cs:
        sess["pn"] += 1
    else:
        sess["pk"] = cs; sess["pn"] = 1
    if sess["pn"] >= _SWITCH_CONFIRM:
        sess["last"] = cur; sess["pk"] = None; sess["pn"] = 0
        return cur
    out = dict(last); out["stale"] = True; return out   # разовый скачок → держим прошлое


# ─────────────────────────── ЧАНКОВЫЙ СТРИМИНГ (WI v2, директива владельца tg_5272) ──────────────
# Старый live/locate декодировал ВЕСЬ РАСТУЩИЙ файл каждый раз → задержка росла, подсветка опаздывала.
# Замеры (work/proto_stream*.py):
#   • disjoint-чанки 4с: декод 48мс, но скелет слишком шумный (wav2vec2 нормализует фичи per-input,
#     4с мало контекста) → k-грамм-локализация прыгает по случайным сурам;
#   • rolling-20с + find_segments по КАЖДОМУ окну: на мелодичном чтении окна дают мусор (улик мало);
#   • find_segments по буферу от НАЧАЛА 30-45с: НАДЁЖНО лочит суру (rec7 → 6:95-98) за ~1с.
# Отсюда дизайн: сервер держит РОЛЛИНГ-буфер PCM, ОДИН раз холодно лочит пассаж find_segments'ом
# (когда накоплено ≥ _COLD_MIN с), затем ведёт позицию SegmentTracker'ом по буквам ВПЕРЁД (монотонно),
# НЕ перезапуская find_segments (на мелодичных окнах он бы прыгнул в мусор). Корпус трекера — от точки
# лока вперёд по Корану (чтение последовательно). Цена константна (окно фиксировано), не растёт.
_STREAM = {}
_ACTIVE = {"n": 0}                                   # число ЖИВЫХ WS-сессий (для idle-деинита модели)
# Простой без единой живой сессии дольше этого → выгружаем wav2vec2 из VRAM (~1.5ГБ) + чистим мёртвые
# сессии. Владелец 30.07: не держать видеопамять, когда никто не читает, чтобы транскрипция large-v3
# получала память БЕЗ перезагрузки сервиса. Во время активного теста модель остаётся (мгновенный ответ).
# 0 → деинит выкл. Таймаут > паузы авто-реконнекта (обрыв ngrok) → короткий разрыв позицию не сбросит.
# 180с (не 45): владелец читает суры ПОДРЯД, останавливаясь между ними на 20-60с. При 45с модель
# успевала выгрузиться в паузе → старт следующей суры перезагружал её (~5с заморозка event-loop →
# «долго думал» + браузер рвёт WS). 3 мин держим тёплой (переключение сур мгновенно), выгружаем лишь
# при РЕАЛЬНОМ простое (для транскрипции large-v3). Крутилка SYNC_LIVE_IDLE_UNLOAD (0 → деинит выкл).
_IDLE_UNLOAD_SEC = float(os.environ.get("SYNC_LIVE_IDLE_UNLOAD", "180"))
_SR = 16000
_BUF_CAP = float(os.environ.get("SYNC_LIVE_BUFCAP", "20"))        # роллинг-буфер PCM (владелец: ~20с, не 48)
_COLD_MIN = float(os.environ.get("SYNC_LIVE_COLDMIN", "10"))      # накопить перед 1-й попыткой лока
_COLD_WIN = float(os.environ.get("SYNC_LIVE_COLDWIN", "20"))      # окно декода для лока/перелока (≤буфера)
# РАСТУЩИЙ cold-аккумулятор для ПЕРВОГО (холодного) лока: роллинг-буфер капнут на _BUF_CAP=20с, но
# find_segments по буферу от НАЧАЛА 30-45с лочит суру НАДЁЖНЕЕ (мусорная сура не держит match на длинном
# окне → её ratio падает, верный пассаж растёт). Копим ВСЁ аудио с начала до _COLD_GROW с ТОЛЬКО в scan;
# на лок используем его вместо роллинга; при переходе в track — освобождаем. 0 → выкл (роллинг как раньше).
_COLD_GROW = float(os.environ.get("SYNC_LIVE_COLDGROW", "45"))
# ⚠️ COLD-CONFIRM (лок при ratio≥floor, если суру подтвердили N раз подряд) ПРОБОВАЛСЯ и ОТВЕРГНУТ:
# устойчивый мусор в окне <20с (rec9 сура 27 «An-Naml» держит ratio 0.43 три лок-попытки подряд, ДО того
# как растущий буфер её давит на n=141) → confirm floor 0.40 лочил суру 27 на t=18с (мислок). Сура-мусор
# rec9 (0.43) сидит в ТОЙ ЖЕ ratio-полосе, что верные пассажи rec5/12 (0.40-0.43) — ratio их не разделяет,
# понижение порога небезопасно. Ускорения холодного лока rec9 нет: узкое место = ratio-гейт + разбавление
# басмалой (короткая Фатиха), не длина буфера. Оставлен только растущий буфер (робастность, см. выше).
_TRACK_WIN = float(os.environ.get("SYNC_LIVE_TRACKWIN", "8"))     # окно декода для трекинга (латентность!)
_FWD_AYAT = int(os.environ.get("SYNC_LIVE_FWD", "120"))           # аятов вперёд в корпусе трекера
_RELOC_STALL = int(os.environ.get("SYNC_LIVE_RELOC", "10"))       # тиков застоя → перелокализация (реже!)
_RELOC_COOLDOWN = int(os.environ.get("SYNC_LIVE_RELOCCD", "20"))  # мин. тиков между перелокализациями
_PROC_STEP = float(os.environ.get("SYNC_LIVE_PROCSTEP", "0.5"))   # с нового аудио между тяж. обработками
_STREAM_MINCHUNK = int(os.environ.get("SYNC_LIVE_MINCHUNK", "800"))
# фаза scan (мульти-гипотеза, растущая уверенность)
_DECAY = float(os.environ.get("SYNC_LIVE_DECAY", "0.6"))          # затухание голосов (недавнее весит больше)
_KTOP = int(os.environ.get("SYNC_LIVE_KTOP", "4"))               # сколько кандидатов показывать
_LOCK_CONF = float(os.environ.get("SYNC_LIVE_LOCKCONF", "0.45")) # доля голосов лидера → точный лок
_LOCK_MIN = float(os.environ.get("SYNC_LIVE_LOCKMIN", "6"))       # мин. буфера перед локом (с)
# ДРОССЕЛЬ холодного лока: тяжёлый decode(до 45с)+find_segments НЕ каждый тик. Пока conf≥порога, но
# ratio ещё не дотянул до гейта (клиент только набирает контекст, ~10-16с), лок-попытка крутилась
# КАЖДЫЙ тик (~0.3-0.5с GPU+difflib с GIL каждые ~0.45с) → event-loop голодал → WS через ngrok рвался
# (owner tg_7171/7172: «сломалось, записалось как две сессии»). Пробуем лок не чаще, чем раз в N тиков.
_COLD_LOCK_N = int(os.environ.get("SYNC_LIVE_COLDLOCKN", "6"))    # тиков между попытками холодного лока
_SCAN_WIN_CH = int(os.environ.get("SYNC_LIVE_SCANWIN", "50"))     # символов декода для голосования
_TRUNC = int(os.environ.get("SYNC_LIVE_TRUNC", "120"))           # обрезка длинного текста аята (симв)
# кросс-сура быстрый перелок в track (фикс холодного мимо-лока): если ДРУГАЯ сура уверенно
# лидирует _RELOCK_HOLD тиков подряд по фоновому scan — перелок сразу, не ждём застоя.
_RELOCK_CONF = float(os.environ.get("SYNC_LIVE_RELOCKCONF", "0.35")) # мин. уверенность кросс-лидера (доля топ-K)
_RELOCK_HOLD = int(os.environ.get("SYNC_LIVE_RELOCKHOLD", "3"))      # тиков подряд лидерства ТОЙ ЖЕ другой суры
# кросс-перелок разрешён ТОЛЬКО в раннем окне после первого лока (фикс ХОЛОДНОГО мимо-лока из-за
# басмалы-интро). Позже — доверяем последовательному треку (легитимная смена суры ловится застоем);
# иначе на мелодичном/повторном чтении k-грамм шумит и кросс прыгал бы по случайным сурам (регресс rec7).
_CROSS_MAX = int(os.environ.get("SYNC_LIVE_CROSSMAX", "40"))         # тиков от первого лока, пока кросс жив
# ГЕЙТ КАЧЕСТВА перелока: свитчим корпус только если предполагаемый пассаж РЕАЛЬНО объясняет декод
# (skeleton-difflib ≥ порога). На мелодичном/шумном декоде find_segments возвращает МУСОРНУЮ суру
# (её ratio низкий) → не прыгаем, остаёмся на месте (последовательность > случайный скачок).
_RELOC_RATIO_MIN = float(os.environ.get("SYNC_LIVE_RELOCRATIO", "0.45"))
# ПОДТВЕРЖДЕНИЕ ПЕРЕЛОКА (фикс мульти-сегмент дрейфа rec9 Фатиха→Исра + ложных интро-локов). Перелок
# ненадёжен одноразово: на ПЕРЕХОДЕ между несмежными пассажами буфер = хвост старого + начало нового +
# такбир → find_segments мелькает МУСОРНОЙ сурой (rec9: 2, 23, 4…), интро-истиаза «فاستعذ بالله» звучит
# как 16:98 (rec12), cross-лидер k-грамм может соврать (rec5 → сура 26). Во всех — ложный кандидат
# мелькает 1 раз, ВЕРНЫЙ держится подряд → берём суру только если тот же кандидат подтвердился
# _RELOC_CONFIRM релок-попыток подряд. Заодно снят блок `len(v2)>=2`: на переходе верная сура приходит
# ОДНОАЯТНЫМ коротким пассажом (len<2) и раньше отвергалась до t=222с.
_RELOC_CONFIRM = int(os.environ.get("SYNC_LIVE_RELOCCONFIRM", "2"))   # релок-попыток подряд той же новой суры
# РЕЖИМ ЗАУЧИВАНИЯ (WI killer-фича, владелец tg_4810): читаешь по памяти, отклонился от текста →
# подсветка КРАСНЫМ. Сигнал = trk.quality (доля свежего декода, покрытая ОЖИДАЕМЫМ окном корпуса
# вокруг указателя). На ВЕРНОМ чтении (даже мелодичном) держится выше _MEM_LOW; при отклонении
# (читают не тот текст) декод перестаёт совпадать с ожиданием → падает. Гистерезис: deviation
# взводится после _MEM_HOLD подряд тиков ниже _MEM_LOW, снимается при quality ≥ _MEM_OK. Работает
# ТОЛЬКО когда клиент включил режим (memorize:1) — в обычном live низкое quality на мелодике/шуме
# = не ошибка чтеца, красным не мигаем.
# ⚠️ КАЛИБРОВКА (перекалибрована 30.07 по TP+FP, work/probe_memorize_tp.py + probe_mem_fp_run.py):
# Старый LOW=0.06 (калибровка probe_memorize по ПРОЦЕНТУ тиков) оказался ПРАКТИЧЕСКИ NO-OP — синтез-
# отклонение (корпус=rec7, кормим ЧУЖОЙ декод rec11) НЕ взводило красный: случайный 3-грамм-оверлап
# арабских согласных держит quality чужого текста ~0.16, ниже 0.06 не падает. Правильная метрика —
# не процент, а МАКС. ПОДРЯД-СЕРИЯ quality<LOW (её и проверяет HOLD). Замер по всем 9 рекам (верное
# чтение, свой декод): худшая ложная серия при LOW≤0.15 = 2-3 тика (< HOLD=5 → НЕ краснит), а чужой
# текст даёт серию ≥5 уже при LOW≥0.08. Окно РАБОЧИХ порогов [0.08..0.15]; при LOW≥0.18 rec10
# (Ар-Рахман) ложно-краснит (серия 8). Дефолт LOW=0.12 — центр окна (худшая своя серия 3, запас 2 до
# HOLD; чужое ловится). Так фича из no-op → реально ловит грубое отклонение (чужой пассаж). ВАЖНО:
# калибровка на МЕЛОДИКЕ = ХУДШИЙ случай; сценарий заучивания = ПРОСТОЕ чтение (rec11/12/13 медиана
# quality 0.75-0.90, ложных серий 0) → запас огромен. Тонкие отклонения (одно неверное слово) слабее
# роняют quality — их порог/чувствительность финально калибрует владелец на своём чтении с телефона.
_MEM_LOW = float(os.environ.get("SYNC_LIVE_MEMLOW", "0.12"))   # ниже → тик «мимо ожидания» (окно [0.08..0.15])
_MEM_OK = float(os.environ.get("SYNC_LIVE_MEMOK", "0.25"))     # ≥ → снять deviation (гистерезис)
_MEM_HOLD = int(os.environ.get("SYNC_LIVE_MEMHOLD", "5"))      # подряд тиков ниже _MEM_LOW → красный


def _trunc(s):
    return s if len(s) <= _TRUNC else s[:_TRUNC].rstrip() + "…"


def _passage_ratio(index, verses, dec):
    """Насколько пассаж (его согласный скелет) объясняет декод — difflib-ratio. Гейт против перелока
    в мусорную суру на шумном/мелодичном декоде (мусор ничего не матчит сильно)."""
    import difflib
    flat, skel = index[3], index[4]
    pos = {sa: i for i, sa in enumerate(flat)}
    txt = "".join(skel[pos[(s, a)]] for (s, a) in verses if (s, a) in pos)
    if not txt or not dec:
        return 0.0
    return difflib.SequenceMatcher(None, dec, txt, autojunk=False).ratio()


def _build_corpus(index, verses):
    """Корпус трекера = пассаж find_segments + продолжение вперёд по Корану (чтение линейно)."""
    flat = index[3]
    fa2i = {fa: i for i, fa in enumerate(flat)}
    corpus = [(s, a) for (s, a) in verses]
    fa_end = fa2i.get((verses[-1][0], verses[-1][1]))
    if fa_end is not None:
        for i in range(fa_end + 1, min(len(flat), fa_end + 1 + _FWD_AYAT)):
            corpus.append((flat[i][0], flat[i][1]))
    return corpus


def _ctx_ayat(q, verses, cur, before=1, after=1):
    """Только предыдущий/текущий/следующий аят (владелец tg_5511: не 5-6, а prev/cur/next), длинные
    обрезаем. current помечаем по позиции в пассаже."""
    ci = next((i for i, v in enumerate(verses) if (v[0], v[1]) == cur), None)
    if ci is None:
        return []
    out = []
    for i in range(max(0, ci - before), min(len(verses), ci + after + 1)):
        s, a = verses[i][0], verses[i][1]
        cur_i = (i == ci)
        txt = _ayah_text(q, s, a)
        out.append({"surah": s, "ayah": a, "text": txt if cur_i else _trunc(txt), "current": cur_i})
    return out


def _decode_window(buf, sec, idx2ch, ch2idx):
    """Декодировать последние `sec` с из PCM-буфера → (E, dec) или (None, '')."""
    import w2v_align
    from match_align import greedy_skeleton
    n = min(len(buf), int(sec * _SR))
    seg = buf[-n:]
    if len(seg) < int(0.5 * _SR):
        return None, ""
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "w.wav")
        import soundfile as sf
        sf.write(wav, seg, _SR)
        E, stride, _i, _c = w2v_align.emissions(wav)
    if E is None or E.shape[0] < 3:
        return None, ""
    special = {ch2idx.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>", "|", "-", "ـ")} - {None}
    return E, greedy_skeleton(E, idx2ch, special, stride_ms=stride)


def _session(sid, reset=False):
    import numpy as np
    st = _STREAM.get(sid)
    if st is None or reset:
        st = _STREAM[sid] = {"buf": np.zeros(0, dtype="float32"),
                             "cold_buf": np.zeros(0, dtype="float32"), "phase": "scan", "n": 0,
                             "verses": None, "trk": None, "votes": None,
                             "last_pos": None, "last_ctx": [], "sent_surah": None,
                             "memorize": False, "dev_low_n": 0, "deviation": False}
    return st


def _build_reply(st, extra):
    q = _quran()
    loc = st.get("last_pos")
    base = {"ok": True, "n": st["n"], "phase": st["phase"], "buf_sec": round(len(st["buf"]) / _SR, 1)}
    if loc:
        base["current"] = {"surah": loc[0], "ayah": loc[1]}
        base["current_text"] = _ayah_text(q, loc[0], loc[1])
        base["ayat"] = st.get("last_ctx", [])
        base["word_frac"] = st.get("word_frac")
        # ВЕСЬ текст суры — только при СМЕНЕ суры (иначе фронт рендерит один раз и лишь двигает
        # подсветку/скролл, без перерисовки). На перелоке (смена пассажа) sent_surah сбросит корпус.
        if st.get("sent_surah") != loc[0]:
            pl = _surah_payload(q, loc[0])
            if pl:
                base["passage"] = pl
                st["sent_surah"] = loc[0]
    if st.get("memorize"):
        base["memorize"] = True
        base["deviation"] = bool(st.get("deviation"))
    base.update(extra)
    return base


def _apply_boost(st, boost):
    """Тап по кандидату (владелец tg_5516): поднять его голоса (затухают общим _DECAY)."""
    import numpy as np
    import match_align
    index = _index(); flat = index[3]
    try:
        bs, ba = boost.split(":")
        fa2i = {fa: i for i, fa in enumerate(flat)}
        fi = fa2i.get((int(bs), int(ba)))
        if fi is not None:
            if st["votes"] is None:
                st["votes"] = np.zeros(len(flat), dtype="float64")
            st["votes"][fi] += float(os.environ.get("SYNC_LIVE_BOOST", "3.0"))
    except Exception:
        pass
    q = _quran()
    cands = match_align.topk_from_votes(st["votes"], index, _KTOP) if st["votes"] is not None else []
    return _build_reply(st, {"candidates": [{"surah": c["surah"], "ayah": c["ayah"],
                        "confidence": c["confidence"], "text": _trunc(_ayah_text(q, c["surah"], c["ayah"]))}
                        for c in cands]})


def _append_pcm(st, pcm):
    """Дёшево (в event-loop): дописать pcm-float в роллинг-буфер + n++. Тяжёлого НЕТ."""
    import numpy as np
    st["buf"] = np.concatenate([st["buf"], pcm])[-int(_BUF_CAP * _SR):]
    # растущий cold-аккумулятор: копим ВСЁ с начала до _COLD_GROW с, ТОЛЬКО пока в scan (до 1-го лока)
    if _COLD_GROW > 0 and st.get("phase") == "scan":
        cb = st.get("cold_buf")
        if cb is not None and len(cb) < int(_COLD_GROW * _SR):
            st["cold_buf"] = np.concatenate([cb, pcm])[-int(_COLD_GROW * _SR):]
    st["n"] += 1


def _process_pcm(st, pcm):
    """HTTP-путь (фолбэк): append + гейт _PROC_STEP + анализ. WS-путь append/анализ вызывает отдельно."""
    _append_pcm(st, pcm)
    new_since = len(st["buf"]) - st.get("last_proc", 0)
    if new_since < int(_PROC_STEP * _SR) and st["n"] > 1:
        return _build_reply(st, {"skip": True})
    return _analyze(st)


def _analyze(st):
    """ТЯЖЁЛОЕ ядро (GPU-декод + scan/track) на ТЕКУЩЕМ буфере — БЕЗ append. В WS зовётся в executor
    single-flight (один разбор за раз на самом свежем буфере) → очередь не растёт, нет бэклога/фриза."""
    import match_align
    from match_align import find_segments, SegmentTracker
    q = _quran(); index = _index(); idx2ch, ch2idx = _vocab()
    try:
        st["last_proc"] = len(st["buf"])
        if st["phase"] == "scan":
            E, dec = _decode_window(st["buf"], _TRACK_WIN, idx2ch, ch2idx)
            if dec:
                dens = match_align.ayah_density(dec[-_SCAN_WIN_CH:], index)
                st["votes"] = dens if st["votes"] is None else st["votes"] * _DECAY + dens
            cands = match_align.topk_from_votes(st["votes"], index, _KTOP) if st["votes"] is not None else []
            if not cands:
                return _build_reply(st, {"scan": True, "candidates": []})
            conf = cands[0]["confidence"]
            cand_out = [{"surah": c["surah"], "ayah": c["ayah"], "confidence": c["confidence"],
                         "text": _trunc(_ayah_text(q, c["surah"], c["ayah"]))} for c in cands]
            cold_step_ok = (st["n"] - st.get("last_coldlock_n", -10**9)) >= _COLD_LOCK_N
            if conf >= _LOCK_CONF and len(st["buf"]) >= int(_LOCK_MIN * _SR) and cold_step_ok:
                st["last_coldlock_n"] = st["n"]
                # растущий аккумулятор (от начала, до 45с) на холодный лок — длинное окно давит мусорную
                # суру и растит ratio верного пассажа; фолбэк на роллинг, если аккумулятор выкл/короче.
                cb = st.get("cold_buf")
                lock_buf = cb if (_COLD_GROW > 0 and cb is not None and len(cb) >= len(st["buf"])) else st["buf"]
                E2, dec2 = _decode_window(lock_buf, max(_COLD_WIN, _COLD_GROW), idx2ch, ch2idx)
                verses = find_segments(E2, q, idx2ch, ch2idx, index=index) if E2 is not None else None
                # ГЕЙТ КАЧЕСТВА ЛОКА: лочимся ТОЛЬКО если пассаж реально объясняет декод (ratio≥порога).
                # Ранний декод (басмала-интро/мелодика) шумный → find_segments даёт МУСОРНУЮ суру с
                # низким ratio → НЕ лочимся, ждём в scan (кандидаты всё равно показываются). Так
                # чинится холодный мимо-лок в КОРНЕ: rec9 ждёт Фатиху (мусор 0.25-0.42 отвергнут,
                # Фатиха 0.5+ принята); Ар-Рахман не встаёт в 29 на 6с (ждёт 55 к ~10с, ratio 0.8).
                lock_ratio = _passage_ratio(index, verses, dec2) if verses else 0.0
                if os.environ.get("SYNC_LIVE_DBG"):
                    print(f"COLDLOCK n={st['n']} buf={len(st['buf'])/_SR:.1f}с v={verses[0] if verses else None}"
                          f" ratio={lock_ratio:.2f} conf={conf:.2f}", flush=True)
                if verses and lock_ratio >= _RELOC_RATIO_MIN:
                    st["verses"] = _build_corpus(index, verses)
                    st["trk"] = SegmentTracker(index, st["verses"])
                    st["phase"] = "track"; st["stall_reloc"] = 0
                    st["cold_buf"] = None           # больше не нужен — освобождаем VRAM/RAM
                    st["lock_n"] = st["n"]          # тик первого лока — окно для кросс-перелока (см. track)
                    _track(st, dec2)
                    return _build_reply(st, {"locked": True, "candidates": cand_out})
            return _build_reply(st, {"scan": True, "candidates": cand_out, "conf": conf})

        # track
        E, dec = _decode_window(st["buf"], _TRACK_WIN, idx2ch, ch2idx)
        moved = _track(st, dec) if dec else False
        cur_surah = st["last_pos"][0] if st.get("last_pos") else None

        # РЕЖИМ ЗАУЧИВАНИЯ: качество совпадения свежего декода с ожидаемым окном корпуса (trk.quality).
        # Падает подряд _MEM_HOLD тиков ниже _MEM_LOW → отклонение (красный); поднимается ≥ _MEM_OK →
        # снимаем (гистерезис). Считаем ВСЕГДА (дёшево), но в reply попадает лишь при memorize:1.
        if st.get("memorize"):
            trk = st.get("trk")
            qy = getattr(trk, "quality", None) if trk is not None else None
            if qy is not None:
                if qy < _MEM_LOW:
                    st["dev_low_n"] = st.get("dev_low_n", 0) + 1
                    if st["dev_low_n"] >= _MEM_HOLD:
                        st["deviation"] = True
                elif qy >= _MEM_OK:
                    st["dev_low_n"] = 0
                    st["deviation"] = False
                if os.environ.get("SYNC_LIVE_DBG"):
                    print(f"MEM n={st['n']} q={qy:.2f} low_n={st['dev_low_n']} dev={st['deviation']}", flush=True)

        # ⚠️ В РЕЖИМЕ ЗАУЧИВАНИЯ корпус ФИКСИРОВАН (читают ИЗВЕСТНЫЙ текст по памяти) → перелок НЕ нужен
        # и ВРЕДЕН: на бедном мелодичном декоде трекер часто застревает → стол-перелок гонял find_segments
        # каждый кулдаун, а это тяжёлый ЧИСТО-PYTHON difflib → держит GIL в executor-потоке ~1с →
        # asyncio event-loop голодает → приём/отдача WS встают → «фронт лёг» (owner 30.07). Возврат сразу,
        # без find_segments: трекер уже прошёл _track выше (позиция/quality/deviation готовы).
        if st.get("memorize"):
            return _build_reply(st, {})

        # Фоновый scan во время track (по УЖЕ декодированному окну — без доп. GPU): ловим ХОЛОДНЫЙ
        # мимо-лок (встали не в ту суру, а трекер фейкает движение по чужому корпусу → застоя нет).
        # Триггерим перелок, только если ДРУГАЯ сура уверенно лидирует _RELOCK_HOLD тиков подряд.
        # Кросс-СУРА → безопасно для рефрена (там лидер = та же сура, не триггерит).
        # ВСЕГДА живые кандидаты в track (владелец tg_7242): держим основную суру, но кандидатов НЕ
        # выкидываем — их вес растёт по мере чтения новой суры, видно на фронте; как только ДРУГАЯ сура
        # уверенно доминирует — свап. Раньше кросс-свап работал лишь в раннем окне (_CROSS_MAX тиков от
        # лока) — «долго думал»/не переключался при смене суры мидстрим. Теперь БЕЗ окна: гард — устойчивое
        # K-подряд лидерство ДРУГОЙ суры + confirm-2 + ratio-гейт на find_segments (рефрен = та же сура,
        # не триггерит; мусорную суру на мелодике давит ratio).
        cross = False
        track_cands = []
        if dec:
            dens = match_align.ayah_density(dec[-_SCAN_WIN_CH:], index)
            st["tvotes"] = dens if st.get("tvotes") is None else st["tvotes"] * _DECAY + dens
            tc = match_align.topk_from_votes(st["tvotes"], index, _KTOP)
            track_cands = [{"surah": c["surah"], "ayah": c["ayah"], "confidence": c["confidence"],
                            "text": _trunc(_ayah_text(q, c["surah"], c["ayah"]))} for c in tc]
            lead_s = tc[0]["surah"] if tc else None
            lead_c = tc[0]["confidence"] if tc else 0.0
            # копим ПОДРЯД тики, где #1 = одна и та же ДРУГАЯ сура (не абсолютный порог — он хрупок и
            # зависит от _KTOP-нормировки); низкий conf-гейт лишь отсекает чистый шум.
            if lead_s is not None and lead_s != cur_surah and lead_c >= _RELOCK_CONF:
                if lead_s == st.get("cross_surah"):
                    st["cross_n"] = st.get("cross_n", 0) + 1
                else:
                    st["cross_surah"] = lead_s; st["cross_n"] = 1
                cross = st["cross_n"] >= _RELOCK_HOLD
            else:
                st["cross_n"] = 0; st["cross_surah"] = None

        if moved and not cross:
            st["stall_reloc"] = 0
        else:
            if not cross:
                st["stall_reloc"] = st.get("stall_reloc", 0) + 1
            cd_ok = (st["n"] - st.get("last_reloc_n", -10**9)) >= _RELOC_COOLDOWN
            stall_trig = st["stall_reloc"] >= _RELOC_STALL and cd_ok       # обычный застой (кулдаун)
            # cross — быстрый путь: без кулдауна (устойчивый K-тиковый лидер уже сильный гард)
            if (cross or stall_trig) and len(st["buf"]) >= int(_COLD_MIN * _SR):
                Er, decr = _decode_window(st["buf"], _COLD_WIN, idx2ch, ch2idx)
                if Er is not None:
                    v2 = find_segments(Er, q, idx2ch, ch2idx, index=index)
                    # ПРИОРИТЕТ ПОСЛЕДОВАТЕЛЬНОСТИ (владелец tg_5744): перелок ТОЛЬКО при смене СУРЫ.
                    # Если та же сура (застой на РЕФРЕНЕ «فبأي آلاء…» ×31) — НЕ перестраиваем корпус
                    # (иначе find_segments выберет другое вхождение того же аята → прыжок по всей суре),
                    # оставляем трекер идти ВПЕРЁД монотонно.
                    # ГЕЙТ КАЧЕСТВА: свитчим, только если новый пассаж реально объясняет декод —
                    # иначе на мелодичном/шумном декоде прыгали бы в МУСОРНУЮ суру (баг rec7-хвоста).
                    ratio = _passage_ratio(index, v2, decr) if v2 else 0.0
                    new_s = v2[0][0] if v2 else None
                    # ЕДИНЫЙ confirm-2 для ЛЮБОГО перелока (cross И stall). Одноразовый find_segments
                    # ненадёжен: (а) на переходном буфере мелькает мусорной сурой (rec9); (б) интро-
                    # истиаза «فاستعذ بالله» звучит как 16:98 → одноразовый лок в суру 16 (rec12);
                    # (в) cross-лидер k-грамм может соврать 3 тика подряд (rec5 → ложная сура 26). Во
                    # всех случаях ЛОЖНЫЙ кандидат мелькает 1 раз, а ВЕРНЫЙ держится подряд → берём суру
                    # только если тот же кандидат подтвердился _RELOC_CONFIRM=2 релок-попытки подряд.
                    # (strong-путь ratio≥0.6 и cross-освобождение УБРАНЫ: впускали одноразовый мусор,
                    # rec9-фикс держится на confirm-2 — там был ratio 0.52<0.6, т.е. strong не участвовал.)
                    qualifies = bool(v2) and new_s != cur_surah and ratio >= _RELOC_RATIO_MIN
                    if qualifies:
                        if new_s == st.get("reloc_cand_s"):
                            st["reloc_cand_n"] = st.get("reloc_cand_n", 0) + 1
                        else:
                            st["reloc_cand_s"] = new_s; st["reloc_cand_n"] = 1
                        confirmed = st["reloc_cand_n"] >= _RELOC_CONFIRM
                    else:
                        confirmed = False
                        st["reloc_cand_s"] = None; st["reloc_cand_n"] = 0
                    if os.environ.get("SYNC_LIVE_DBG"):
                        print(f"RELOC n={st['n']} cross={cross} stall={stall_trig} cur={cur_surah} "
                              f"v2={v2[0] if v2 else None} ratio={ratio:.2f} "
                              f"cand={st.get('reloc_cand_s')}×{st.get('reloc_cand_n')} conf={confirmed}", flush=True)
                    if confirmed:
                        st["verses"] = _build_corpus(index, v2)
                        st["trk"] = SegmentTracker(index, st["verses"])
                        st["tvotes"] = None
                        st["reloc_cand_s"] = None; st["reloc_cand_n"] = 0
                        _track(st, decr)
                st["last_reloc_n"] = st["n"]
                st["stall_reloc"] = 0
                st["cross_n"] = 0; st["cross_surah"] = None  # после попытки — копим заново (бережём GPU)
        return _build_reply(st, {"candidates": track_cands})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("LIVE_PROCESS_ERROR:\n" + tb, flush=True)
        return _build_reply(st, {"ok": False, "error": type(e).__name__ + ": " + str(e), "trace": tb[-600:]})


def _pcm_from_raw(raw, is_pcm):
    """Байты чанка → np.float32 16кГц моно (сырой Int16 или webm/opus через ffmpeg). None если пусто."""
    import numpy as np
    import w2v_align
    if is_pcm:
        return np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "c.bin"); wav = os.path.join(d, "c.wav")
        with open(src, "wb") as f:
            f.write(raw)
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", src, "-ar", str(_SR), "-ac", "1", wav],
                       capture_output=True)
        if not (os.path.exists(wav) and os.path.getsize(wav) > 4000):
            return None
        return w2v_align._load_wav(wav)


@csrf_exempt
def live_stream(request):
    """POST ?sid=&reset=&pcm=&boost= — HTTP-обёртка над общим ядром (фолбэк, если нет вебсокета)."""
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    sid = request.GET.get("sid", "") or "_"
    st = _session(sid, reset=bool(request.GET.get("reset")))
    boost = request.GET.get("boost")
    if boost:
        return JsonResponse(_apply_boost(st, boost))
    raw = request.body
    is_pcm = bool(request.GET.get("pcm")) or "octet" in (request.content_type or "")
    if not raw or (not is_pcm and len(raw) < _STREAM_MINCHUNK) or (is_pcm and len(raw) < 320):
        return JsonResponse(_build_reply(st, {"empty": True}))
    pcm = _pcm_from_raw(raw, is_pcm)
    if pcm is None:
        return JsonResponse(_build_reply(st, {"empty": True, "detail": "no audio"}))
    return JsonResponse(_process_pcm(st, pcm))


def _track(st, dec):
    """Скормить декод трекеру окнами; обновить last_pos/last_ctx. Вернуть True, если позиция сдвинулась."""
    q = _quran()
    trk = st["trk"]
    prev = st.get("last_pos")
    cur = None
    W = 40
    # ОДНО кормление свежего хвоста за тик (континуитет, указка владельца): раньше цикл прогонял
    # ВСЁ окно _TRACK_WIN (5-10 feed'ов за тик) → указатель продвигался многократно → РАЗГОН вперёд
    # (racing на рефрене). Человек читает последовательно — за тик продвигаемся на свежий кусок раз.
    cur = trk.feed(dec[-W:]) if dec else None
    if cur is None and st["verses"]:
        cur = {"surah": st["verses"][0][0], "ayah": st["verses"][0][1]}
    if cur:
        st["last_pos"] = (cur["surah"], cur["ayah"])
        st["last_ctx"] = _ctx_ayat(q, st["verses"], (cur["surah"], cur["ayah"]))
        # доля пройденного ТЕКУЩЕГО аята по позиции трекера в M → пословная подсветка на фронте
        # (владелец: пословные тайминги не выкидываем, подсвечивать слово «по возможности»)
        try:
            sa, p = trk.sa, min(trk.p, len(trk.sa) - 1)
            key = sa[p]
            a0 = a1 = p
            while a0 > 0 and sa[a0 - 1] == key:
                a0 -= 1
            while a1 < len(sa) - 1 and sa[a1 + 1] == key:
                a1 += 1
            st["word_frac"] = round((p - a0) / max(1, a1 - a0), 3)
        except Exception:
            st["word_frac"] = None
    return st.get("last_pos") != prev


def _do_locate(raw: bytes):
    import numpy as np  # noqa
    import match_align
    import w2v_align
    from match_align import greedy_skeleton, find_segments, SegmentTracker

    q = _quran()
    index = _index()
    idx2ch, ch2idx = _vocab()

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.bin")
        wav = os.path.join(d, "a.wav")
        with open(src, "wb") as f:
            f.write(raw)
        # последние _TAIL_SEC с (свежий контекст, ограничение латентности) → wav 16кГц моно.
        # -sseof на НЕЗАВЕРШЁННОМ webm с микрофона может дать битый/пустой файл → всегда пробуем и
        # ПОЛНУЮ конвертацию, берём тот wav, что реально содержит аудио (по размеру).
        wav2 = os.path.join(d, "b.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-sseof", f"-{_TAIL_SEC}", "-i", src,
                        "-ar", "16000", "-ac", "1", wav], capture_output=True)
        rf = subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", src, "-ar", "16000", "-ac", "1", wav2],
                            capture_output=True)
        cand = [p for p in (wav, wav2) if os.path.exists(p) and os.path.getsize(p) > 4000]
        if not cand:
            return JsonResponse({"ok": True, "empty": True,
                                 "detail": rf.stderr.decode("utf-8", "ignore")[-300:]})
        use = min(cand, key=os.path.getsize)   # sseof-хвост если валиден, иначе полный
        E, stride, _i2c, _c2i = w2v_align.emissions(use)

    if E is None or E.shape[0] < 3:
        return JsonResponse({"ok": True, "empty": True})

    special = {ch2idx.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>", "|", "-", "ـ")} - {None}
    dec = greedy_skeleton(E, idx2ch, special, stride_ms=stride)
    # слишком мало букв (2-3с зачина/муqаттаʿат) → локализация шаткая (даёт ложный الر и т.п.);
    # ждём накопления контекста, чтобы не давать неверную ПЕРВУЮ привязку (её потом держит липкость)
    _MIN_DEC = int(os.environ.get("SYNC_LIVE_MINDEC", "35"))
    if len(dec) < _MIN_DEC:
        return JsonResponse({"ok": True, "empty": True, "warmup": True, "decode": dec})

    verses = find_segments(E, q, idx2ch, ch2idx, index=index)
    if not verses:
        return JsonResponse({"ok": True, "found": False, "decode": dec[-80:]})

    # текущая позиция внутри пассажа — SegmentTracker по всему свежему декоду
    trk = SegmentTracker(index, verses)
    cur = None
    W = 40
    step = 20
    for end in range(min(W, len(dec)), len(dec) + 1, step):
        cur = trk.feed(dec[max(0, end - W):end])
    if cur is None and verses:
        cur = {"surah": verses[-1][0], "ayah": verses[-1][1]}

    # сегменты в компактную запись + собрать текст пассажа (с пометкой текущего аята)
    segs = []
    s0 = 0
    for i in range(1, len(verses) + 1):
        brk = (i == len(verses)) or verses[i][0] != verses[i - 1][0] or verses[i][1] != verses[i - 1][1] + 1
        if brk:
            a, b = verses[s0], verses[i - 1]
            segs.append({"surah": a[0], "ayah_start": a[1], "ayah_end": b[1]})
            s0 = i
    ayat = [{"surah": s, "ayah": a, "text": _ayah_text(q, s, a),
             "current": (cur is not None and s == cur["surah"] and a == cur["ayah"])}
            for (s, a) in verses]

    return JsonResponse({
        "ok": True, "found": True,
        "segments": segs,
        "current": cur,
        "current_text": _ayah_text(q, cur["surah"], cur["ayah"]) if cur else "",
        "ayat": ayat,
        "decode_tail": dec[-80:],
        "n_verses": len(verses),
    })


# ─────────────────────────── WEBSOCKET (WI, директива владельца tg_5373/5558) ───────────────────
# Причина зависания HTTP-версии: 100мс-чанки = ~10 запросов/с, ngrok-free троттлит и рвёт коннект.
# Вебсокет — ОДИН постоянный коннект: кадры PCM текут потоком, сервер шлёт JSON-обновления, нет
# пер-запросного лимита. Тяжёлый декод (GPU, блокирующий) гоняем в threadpool, чтобы не держать loop.
async def _idle_deinit(loop):
    """После простоя _IDLE_UNLOAD_SEC без живых сессий — выгрузить модель из VRAM + подчистить мёртвые
    сессии. Планируется при падении числа сессий до 0; повторно проверяет счётчик (быстрый реконнект
    его поднимет → деинит отменится). Безопасно: при 0 сессий декод не идёт (unload не рвёт inference)."""
    import asyncio
    if _IDLE_UNLOAD_SEC <= 0:
        return
    await asyncio.sleep(_IDLE_UNLOAD_SEC)
    if _ACTIVE["n"] > 0:
        return                                        # кто-то подключился за время простоя → не трогаем
    try:
        import w2v_align
        freed = await loop.run_in_executor(None, w2v_align.unload)
        _STREAM.clear()                               # живых сессий нет → состояние трекеров/буферов не нужно
        if freed:
            print("LIVE idle → wav2vec2 выгружен из VRAM, сессии очищены", flush=True)
    except Exception:
        pass


async def live_ws(scope, receive, send):
    """SINGLE-FLIGHT: приём кадров дёшево в loop (_append_pcm), тяжёлый _analyze — ОДИН за раз в
    executor на самом свежем буфере (кадры между разборами просто продлевают буфер) → очередь НЕ
    растёт, нет бэклога/фриза. Плюс: пишем СЫРОЙ поток владельца в файл (дебаг на его аудио) + лог
    подключений (видно в мониторе, когда он стримит)."""
    import asyncio
    import json as _json
    import numpy as np
    from urllib.parse import parse_qs
    qs = parse_qs((scope.get("query_string") or b"").decode())
    sid = (qs.get("sid", ["_"])[0]) or "_"
    # reset ТОЛЬКО если явно (?reset=1) или сессии ещё нет → при авто-реконнекте позиция СОХРАНЯЕТСЯ
    # (обрыв ngrok не сбрасывает в ре-скан → нет прыжков «с нуля», владелец tg_5704/5708)
    st = _session(sid, reset=bool(qs.get("reset")) or sid not in _STREAM)
    # generation-guard: при обрыве ngrok сервер НЕ узнаёт, что клиент ушёл (ws-ping-interval=0 → мёртвый
    # коннект не реапится), клиент открывает НОВЫЙ WS того же sid → два хэндлера живут разом и гонятся за
    # общий st (буфер/фаза/трекер) → состояние бьётся, «вернулось в scan» (owner tg_7185, две сессии в
    # логах на один sid без disconnect между). Новый коннект вытесняет старый: каждый хэндлер помнит свой
    # gen; как только gen сменился (пришёл новый) — старый (зомби) тихо выходит, не трогая общий st.
    st["gen"] = st.get("gen", 0) + 1
    my_gen = st["gen"]
    loop = asyncio.get_event_loop()
    # запись сырого потока (Int16 16кГц) — конвертнуть в wav: ffmpeg -f s16le -ar 16000 -ac 1 -i <f> out.wav
    cap_path = os.path.join(os.environ.get("SYNC_WORK", "/app/work"), f"live_cap_{sid}.pcm")
    try:
        cap = open(cap_path, "wb")
    except Exception:
        cap = None
    print(f"LIVE_WS connect sid={sid} peer={scope.get('client')} → запись {cap_path}", flush=True)
    await send({"type": "websocket.accept"})
    _ACTIVE["n"] += 1                                 # живая сессия (для idle-деинита модели)
    # ПРЕЛОАД модели в фоне, пока клиент набирает первые ~6с буфера: если модель была выгружена
    # (простой) — грузим её ПАРАЛЛЕЛЬНО набору контекста, чтобы первый декод не морозил loop на ~5с
    # посреди стрима (это и рвало WS + давало «долго думал»). Если уже тёплая — no-op.
    try:
        import w2v_align as _w2v
        if not _w2v.is_loaded():
            loop.run_in_executor(None, _w2v.warmup)
    except Exception:
        pass

    proc = {"task": None, "dirty": False, "frames": 0}

    async def pump():
        while proc["dirty"]:
            if st.get("gen") != my_gen:                              # вытеснён новым коннектом → не анализируем
                return
            proc["dirty"] = False
            res = await loop.run_in_executor(None, _analyze, st)     # GPU вне loop, single-flight
            try:
                await send({"type": "websocket.send", "text": _json.dumps(res)})
            except Exception:
                return

    try:
        while True:
            ev = await receive()
            if st.get("gen") != my_gen:                  # новый коннект того же sid вытеснил нас → выходим
                break
            t = ev.get("type")
            if t == "websocket.disconnect":
                break
            if t != "websocket.receive":
                continue
            if ev.get("bytes") is not None:              # кадр сырого Int16 PCM 16кГц
                raw = ev["bytes"]
                if len(raw) < 320:
                    continue
                if cap:
                    try: cap.write(raw)
                    except Exception: pass
                proc["frames"] += 1
                if proc["frames"] % 50 == 0:
                    print(f"LIVE_WS sid={sid} frames={proc['frames']} buf={len(st['buf'])/_SR:.1f}с phase={st['phase']}", flush=True)
                _append_pcm(st, np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0)
                proc["dirty"] = True
                if proc["task"] is None or proc["task"].done():
                    proc["task"] = asyncio.create_task(pump())       # запустить разбор, если не идёт
            elif ev.get("text") is not None:             # управление: reset / boost:s:a
                msg = ev["text"].strip()
                if msg == "reset":
                    st = _session(sid, reset=True)
                elif msg.startswith("memorize:"):     # режим заучивания вкл/выкл (красное отклонение)
                    st["memorize"] = msg.endswith("1")
                    if not st["memorize"]:
                        st["deviation"] = False; st["dev_low_n"] = 0
                elif msg.startswith("boost:"):
                    res = await loop.run_in_executor(None, _apply_boost, st, msg[6:])
                    await send({"type": "websocket.send", "text": _json.dumps(res)})
    except Exception:
        import traceback
        print("LIVE_WS_ERROR:\n" + traceback.format_exc(), flush=True)
    finally:
        if cap:
            try: cap.close()
            except Exception: pass
        _ACTIVE["n"] = max(0, _ACTIVE["n"] - 1)
        print(f"LIVE_WS disconnect sid={sid} frames={proc['frames']} active={_ACTIVE['n']}", flush=True)
        if _ACTIVE["n"] == 0:                         # никого не осталось → запланировать выгрузку модели
            try:
                import asyncio
                asyncio.create_task(_idle_deinit(loop))
            except Exception:
                pass
    try:
        await send({"type": "websocket.close"})
    except Exception:
        pass
