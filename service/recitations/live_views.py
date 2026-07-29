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
        return _sticky(sid, {"ok": True, "empty": True})

    try:
        res = _do_locate(raw)
        # липкость: если сейчас ничего не нашли, но раньше находили — вернём прошлое (антимерцание)
        import json as _json
        payload = _json.loads(res.content)
        if payload.get("found"):
            _SESS[sid] = payload
            return res
        return _sticky(sid, payload)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("LIVE_LOCATE_ERROR:\n" + tb, flush=True)   # видно в логах (монитор)
        return _sticky(sid, {"ok": False, "error": type(e).__name__ + ": " + str(e),
                             "trace": tb[-900:]})


def _sticky(sid, cur):
    """Вернуть текущий ответ, либо последний НАЙДЕННЫЙ для сессии с пометкой stale (если сейчас пусто)."""
    last = _SESS.get(sid)
    if not cur.get("found") and last:
        out = dict(last)
        out["stale"] = True
        return JsonResponse(out)
    return JsonResponse(cur)


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
