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
_TAIL_SEC = float(os.environ.get("SYNC_LIVE_TAIL", "25"))   # берём последние N с аудио (свежий контекст)


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
    raw = request.body
    if not raw or len(raw) < 2000:
        return JsonResponse({"ok": True, "empty": True})

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
        # последние _TAIL_SEC с (свежий контекст, ограничение латентности) → wav 16кГц моно
        cmd = ["ffmpeg", "-nostdin", "-y", "-sseof", f"-{_TAIL_SEC}", "-i", src,
               "-ar", "16000", "-ac", "1", wav]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0 or not os.path.exists(wav):   # -sseof не сработал (аудио короче) → без него
            r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", src, "-ar", "16000", "-ac", "1", wav],
                               capture_output=True)
        if r.returncode != 0 or not os.path.exists(wav):
            return JsonResponse({"error": "ffmpeg", "detail": r.stderr.decode("utf-8", "ignore")[-500:]},
                                status=500)

        E, stride, _i2c, _c2i = w2v_align.emissions(wav)

    special = {ch2idx.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>", "|", "-", "ـ")} - {None}
    dec = greedy_skeleton(E, idx2ch, special, stride_ms=stride)
    if len(dec) < 5:
        return JsonResponse({"ok": True, "empty": True, "decode": dec})

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
