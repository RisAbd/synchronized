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
_SR = 16000
_BUF_CAP = float(os.environ.get("SYNC_LIVE_BUFCAP", "48"))        # сколько с PCM держим
_COLD_MIN = float(os.environ.get("SYNC_LIVE_COLDMIN", "22"))      # накопить перед 1-й попыткой лока
_COLD_WIN = float(os.environ.get("SYNC_LIVE_COLDWIN", "45"))      # окно декода для холодного лока
_TRACK_WIN = float(os.environ.get("SYNC_LIVE_TRACKWIN", "14"))    # окно декода для трекинга
_FWD_AYAT = int(os.environ.get("SYNC_LIVE_FWD", "120"))           # аятов вперёд в корпусе трекера
_RELOC_STALL = int(os.environ.get("SYNC_LIVE_RELOC", "4"))        # чанков застоя → перелокализация
_STREAM_MINCHUNK = int(os.environ.get("SYNC_LIVE_MINCHUNK", "800"))


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


def _ctx_ayat(q, verses, cur, before=1, after=4):
    """Соседние аяты вокруг текущего в ПАССАЖЕ чтения (контекст «память рядом»), с пометкой current."""
    ci = next((i for i, v in enumerate(verses) if (v[0], v[1]) == cur), None)
    if ci is None:
        return []
    out = []
    for i in range(max(0, ci - before), min(len(verses), ci + after + 1)):
        s, a = verses[i][0], verses[i][1]
        # текущий = cur — если чтец перечитывает, cur может совпасть с несколькими; помечаем по позиции
        out.append({"surah": s, "ayah": a, "text": _ayah_text(q, s, a), "current": (i == ci)})
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


@csrf_exempt
def live_stream(request):
    """POST: тело = КОРОТКИЙ самостоятельный аудио-чанк (webm/opus/wav, ~4с). ?sid=…&reset=1 сброс.
    Сервер копит PCM-буфер, холодно лочит пассаж, ведёт позицию трекером. Ответ: место + контекст."""
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    import numpy as np
    import match_align
    from match_align import find_segments, SegmentTracker
    import w2v_align
    q = _quran()
    index = _index()
    flat = index[3]
    idx2ch, ch2idx = _vocab()
    sid = request.GET.get("sid", "") or "_"

    st = _STREAM.get(sid)
    if st is None or request.GET.get("reset"):
        st = _STREAM[sid] = {"buf": np.zeros(0, dtype="float32"), "phase": "cold", "n": 0,
                             "verses": None, "trk": None, "cold_surah": None, "cold_hits": 0,
                             "last_pos": None, "last_ctx": []}

    def _reply(extra):
        loc = st.get("last_pos")
        base = {"ok": True, "n": st["n"], "phase": st["phase"],
                "buf_sec": round(len(st["buf"]) / _SR, 1)}
        if loc:
            base["current"] = {"surah": loc[0], "ayah": loc[1]}
            base["current_text"] = _ayah_text(q, loc[0], loc[1])
            base["ayat"] = st.get("last_ctx", [])
        base.update(extra)
        return JsonResponse(base)

    raw = request.body
    if not raw or len(raw) < _STREAM_MINCHUNK:
        return _reply({"empty": True})

    try:
        # чанк webm/opus → PCM 16к, дописать в роллинг-буфер (склейка PCM без webm-заголовков)
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "c.bin"); wav = os.path.join(d, "c.wav")
            with open(src, "wb") as f:
                f.write(raw)
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", src, "-ar", str(_SR), "-ac", "1", wav],
                           capture_output=True)
            if not (os.path.exists(wav) and os.path.getsize(wav) > 4000):
                return _reply({"empty": True, "detail": "no audio in chunk"})
            pcm = w2v_align._load_wav(wav)
        st["buf"] = np.concatenate([st["buf"], pcm])[-int(_BUF_CAP * _SR):]
        st["n"] += 1

        if st["phase"] == "cold":
            if len(st["buf"]) < int(_COLD_MIN * _SR):
                return _reply({"warmup": True, "buf_need": _COLD_MIN})
            E, dec = _decode_window(st["buf"], _COLD_WIN, idx2ch, ch2idx)
            if E is None:
                return _reply({"warmup": True})
            verses = find_segments(E, q, idx2ch, ch2idx, index=index)
            if not verses or len(verses) < 2:
                return _reply({"cold": True, "seg": 0})
            surah0 = verses[0][0]
            # континуитет: та же сура 2 попытки подряд → лочим (защита от разового ложного окна)
            if st["cold_surah"] == surah0:
                st["cold_hits"] += 1
            else:
                st["cold_surah"] = surah0; st["cold_hits"] = 1
            if st["cold_hits"] < 2:
                return _reply({"cold": True, "seg": len(verses), "cand": f"{surah0}:{verses[0][1]}"})
            # ЛОК: корпус трекера = пассаж find_segments + продолжение вперёд по Корану
            st["verses"] = _build_corpus(index, verses)
            st["trk"] = SegmentTracker(index, st["verses"])
            st["phase"] = "track"
            st["stall_reloc"] = 0
            _track(st, dec)                                 # сразу протрекать по накопленному декоду
            return _reply({"locked": True, "seg": len(verses)})

        # phase == track: декодим последние _TRACK_WIN с → ведём позицию по хвосту
        E, dec = _decode_window(st["buf"], _TRACK_WIN, idx2ch, ch2idx)
        moved = _track(st, dec) if dec else False
        # ПЕРЕЛОКАЛИЗАЦИЯ по застреванию: трекер не двигается _RELOC_STALL чанков → корпус неверен
        # (сменился пассаж / мульти-сегмент Фатиха→Исра / ошибочный первичный лок). find_segments по
        # роллинг-буферу надёжен (не мелодичное окно) → пересобираем корпус и переискиваем позицию.
        if moved:
            st["stall_reloc"] = 0
        else:
            st["stall_reloc"] = st.get("stall_reloc", 0) + 1
            if st["stall_reloc"] >= _RELOC_STALL and len(st["buf"]) >= int(_COLD_MIN * _SR):
                Er, decr = _decode_window(st["buf"], _COLD_WIN, idx2ch, ch2idx)
                if Er is not None:
                    v2 = find_segments(Er, q, idx2ch, ch2idx, index=index)
                    if v2 and len(v2) >= 2:
                        st["verses"] = _build_corpus(index, v2)
                        st["trk"] = SegmentTracker(index, st["verses"])
                        _track(st, decr)                    # переискать текущую позицию
                st["stall_reloc"] = 0
        return _reply({})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("LIVE_STREAM_ERROR:\n" + tb, flush=True)
        return _reply({"ok": False, "error": type(e).__name__ + ": " + str(e), "trace": tb[-600:]})


def _track(st, dec):
    """Скормить декод трекеру окнами; обновить last_pos/last_ctx. Вернуть True, если позиция сдвинулась."""
    q = _quran()
    trk = st["trk"]
    prev = st.get("last_pos")
    cur = None
    W = 40
    for end in range(min(W, len(dec)), len(dec) + 1, 20):
        cur = trk.feed(dec[max(0, end - W):end])
    if cur is None and st["verses"]:
        cur = {"surah": st["verses"][0][0], "ayah": st["verses"][0][1]}
    if cur:
        st["last_pos"] = (cur["surah"], cur["ayah"])
        st["last_ctx"] = _ctx_ayat(q, st["verses"], (cur["surah"], cur["ayah"]))
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
