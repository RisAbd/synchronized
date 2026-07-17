"""wav2vec2 forced alignment (через whisperx) — альтернатива MMS-forced (`falign.py`).

Зачем: MMS ctc-forced-aligner романизует арабский и СХЛОПЫВАЕТ огласовки → тянущейся гласной
(мадд) не даёт токена, конец слова садится на согласный спайк, а протяжка падает в «дыру».
На мелодичном таджвиде это (а) роняет подсветку слова раньше времени, (б) занижает coverage
(«речь» ~0.5, хотя запись почти сплошь звучит). wav2vec2 (CTC поверх сырых фонем, без романизации)
держит слово сквозь мадд → границы честнее, coverage считается по настоящим t_end.

Выдаёт sync_map ТОЙ ЖЕ формы, что `falign.align`: {meta, timeline, word_timeline, char_timeline},
совместимой с `player.build_data`. Вход тот же: verses=[(surah, ayah, text), ...].

Длинное аудио бьём на короткие сегменты (память wav2vec2 линейна по длине окна; один кусок на всё
падает OOM на 6ГБ уже с ~250с). Границы окон — грубые тайминги аятов из ASR-источника. Чтобы
слово у СТЫКА окон не прижималось к обрезанному краю, каждый сегмент выравниваем с КОНТЕКСТОМ ±1 аят
и берём тайминг слова из того окна, где оно ДАЛЬШЕ от края (interior-pick) — так граничные слова
всегда выровнены с опорой с обеих сторон.

Запуск ТОЛЬКО на GPU (правило проекта). Живёт в отдельном venv (~/.venvs/whisperx, torch cu128),
whisperx импортируется лениво — модуль можно импортировать где угодно (available() проверит deps).
"""
from __future__ import annotations

import difflib
import os

import falign          # только за хелперами (_snap_bounds, _HARAKAT); тяжёлые импорты у него ленивы
import quran as quranmod

_HARAKAT = falign._HARAKAT
SAMPLE_RATE = 16000

_WINDOW_PAD = 2.0        # запас аудио с каждого края окна (с) — контекст, не лишний текст
_MAX_WINDOW_SEC = 25.0   # макс длительность CORE окна → пиковая память wav2vec2 ограничена
_CTX_AYAT = 1            # сколько аятов контекста добавлять с каждой стороны (overlap)

_ALIGN_MODEL = os.environ.get("SYNC_W2V_MODEL", "")   # пусто → дефолтная арабская whisperx
_align_model = None
_align_meta = None


def available() -> bool:
    """Есть ли whisperx+torch в текущем окружении (без загрузки моделей)."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("whisperx", "torch"))


def _norm(w: str) -> str:
    return quranmod.normalize(w)


def _load_model(device: str):
    global _align_model, _align_meta
    if _align_model is None:
        import whisperx
        if _ALIGN_MODEL:
            _align_model, _align_meta = whisperx.load_align_model(
                language_code="ar", device=device, model_name=_ALIGN_MODEL)
        else:
            _align_model, _align_meta = whisperx.load_align_model(language_code="ar", device=device)
    return _align_model, _align_meta


def _fill_starts(windows, verses, dur):
    """Недостающие старты аятов — линейной интерполяцией по накопленному числу слов между якорями."""
    n = len(verses)
    nwords = [len(v[2].split()) for v in verses]
    cum = [0]
    for k in nwords:
        cum.append(cum[-1] + k)
    starts = [w[0] if w and w[0] is not None else None for w in windows]
    known = [i for i, s in enumerate(starts) if s is not None]
    if not known:
        total = cum[-1] or 1
        return [dur * cum[i] / total for i in range(n)]
    out = [None] * n
    for i in range(n):
        if starts[i] is not None:
            out[i] = float(starts[i]); continue
        left = max([k for k in known if k < i], default=None)
        right = min([k for k in known if k > i], default=None)
        if left is not None and right is not None:
            f = (cum[i] - cum[left]) / max(1, cum[right] - cum[left])
            out[i] = starts[left] + (starts[right] - starts[left]) * f
        elif left is not None:
            f = (cum[i] - cum[left]) / max(1, cum[-1] - cum[left])
            out[i] = starts[left] + (dur - starts[left]) * f
        else:
            out[i] = starts[right] * (cum[i] / max(1, cum[right]))
    return out


def _build_segments(verses, starts, dur, vranges):
    """Сегменты ≤ _MAX_WINDOW_SEC с контекстом ±_CTX_AYAT аята (overlap).

    vranges — [(rstart, rend), ...] диапазон ref-индексов слов на каждый аят.
    Возвращает [(win_start, win_end, text, ref_indices)]: ref_indices — плоские индексы ref-слов
    в порядке текста, ВКЛЮЧАЯ контекст (по ним потом маплем обратно и выбираем interior).
    """
    n = len(verses)
    segs = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and (starts[j + 1] - starts[i]) < _MAX_WINDOW_SEC:
            j += 1
        ci = max(0, i - _CTX_AYAT)
        cj = min(n - 1, j + _CTX_AYAT)
        win_start = max(0.0, starts[ci] - _WINDOW_PAD)
        win_end = min(dur, (starts[cj + 1] if cj + 1 < n else dur) + _WINDOW_PAD)
        text = " ".join(verses[k][2] for k in range(ci, cj + 1))
        ref_idx = list(range(vranges[ci][0], vranges[cj][1]))
        segs.append((win_start, win_end, text, ref_idx))
        i = j + 1
    return segs


def align(audio_path, verses, windows=None, snap: bool | None = None) -> dict:
    """wav2vec2 forced alignment диапазона аятов к аудио. GPU обязателен.

    verses — [(surah, ayah, text), ...]. windows — [[start,end], ...] по verses (грубые тайминги
    аятов из ASR для нарезки; None → один сегмент, только для коротких записей).
    """
    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("w2v_align требует GPU (torch.cuda недоступна) — CPU не гоняем")

    # выкинуть токены-вакфы/паузы (знаки-неслова) из текста аятов — единая безвакфовая индексация
    # wi (как align.py/build_data); дальше весь пайплайн (ref/сегменты/счётчики) на очищенном тексте.
    verses = [(s, a, " ".join(quranmod.word_tokens(t))) for s, a, t in verses]

    # ref: плоский список слов + диапазоны индексов на аят
    ref = []
    vranges = []
    for surah, ayah, txt in verses:
        r0 = len(ref)
        for wi, w in enumerate(txt.split()):
            ref.append((surah, ayah, wi, w))
        vranges.append((r0, len(ref)))
    ref_words = [r[3] for r in ref]

    audio = whisperx.load_audio(str(audio_path))
    dur = len(audio) / SAMPLE_RATE
    model_a, meta_a = _load_model(device)

    if windows:
        starts = _fill_starts(windows, verses, dur)
        segments = _build_segments(verses, starts, dur, vranges)
    else:
        segments = [(0.0, dur, " ".join(v[2] for v in verses), list(range(len(ref))))]

    seg_inputs = [{"start": s, "end": e, "text": t} for s, e, t, _ in segments]
    res = whisperx.align(seg_inputs, model_a, meta_a, audio, device, return_char_alignments=False)
    torch.cuda.empty_cache()
    res_segments = res.get("segments", [])

    # кандидаты на каждый ref-индекс: (t, t_end, edge_dist). Берём с макс edge_dist (interior).
    best = {}  # gi -> (t, t_end, edge_dist)
    for k, (win_s, win_e, _text, ref_idx) in enumerate(segments):
        if k >= len(res_segments):
            break
        wx = [w for w in (res_segments[k].get("words") or [])
              if w.get("start") is not None and w.get("end") is not None]
        seg_ref_norm = [_norm(ref_words[gi]) for gi in ref_idx]
        wx_norm = [_norm(w.get("word", "")) for w in wx]
        sm = difflib.SequenceMatcher(a=seg_ref_norm, b=wx_norm, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("equal", "replace"):
                for d in range(min(i2 - i1, j2 - j1)):
                    gi = ref_idx[i1 + d]
                    w = wx[j1 + d]
                    s, e = float(w["start"]), float(w["end"])
                    if e <= s:
                        continue
                    mid = (s + e) / 2
                    edge = min(mid - win_s, win_e - mid)
                    prev = best.get(gi)
                    if prev is None or edge > prev[2]:
                        best[gi] = (s, e, edge)

    bounds_opt = [None] * len(ref)
    for gi, (s, e, _edge) in best.items():
        bounds_opt[gi] = (s, e)
    matched = sum(1 for b in bounds_opt if b is not None)

    bounds, interp_flags = _interp_missing(bounds_opt)

    # подтяжка границ к тишине (RMS), только у реально выровненных (опт-аут SYNC_W2V_SNAP=0)
    snapped = 0
    do_snap = (os.environ.get("SYNC_W2V_SNAP", "1") != "0") if snap is None else snap
    if do_snap:
        real_idx = [i for i, f in enumerate(interp_flags) if not f]
        real_bounds = [bounds[i] for i in real_idx]
        snapped_bounds, snapped = falign._snap_bounds(real_bounds, audio)
        for k, i in enumerate(real_idx):
            bounds[i] = snapped_bounds[k]

    # строгий рост t + сборка дорожек
    word_timeline, timeline, char_timeline = [], [], []
    seen_ayah = set()
    for i, (surah, ayah, wi, arabic) in enumerate(ref):
        t0, t1 = bounds[i]
        entry = {"t": round(t0, 3), "surah": surah, "ayah": ayah, "wi": wi}
        if not interp_flags[i] and t1 > t0:
            entry["t_end"] = round(t1, 3)
        word_timeline.append(entry)
        if (surah, ayah) not in seen_ayah:
            seen_ayah.add((surah, ayah))
            timeline.append({"t": round(t0, 3), "surah": surah, "ayah": ayah})
        if t1 > t0:
            base_positions = [p for p, ch in enumerate(arabic) if ch not in _HARAKAT]
            nb = max(1, len(base_positions))
            for ci, ch in enumerate(arabic):
                kk = sum(1 for p in base_positions if p < ci)
                frac0 = kk / nb
                frac1 = (kk + 1) / nb
                ct0 = t0 + (t1 - t0) * frac0
                ct1 = t0 + (t1 - t0) * (frac0 if ch in _HARAKAT else frac1)
                char_timeline.append({"t": round(ct0, 3), "t_end": round(ct1, 3),
                                      "surah": surah, "ayah": ayah, "wi": wi, "ci": ci})
    for i in range(1, len(word_timeline)):
        if word_timeline[i]["t"] <= word_timeline[i - 1]["t"]:
            word_timeline[i]["t"] = round(word_timeline[i - 1]["t"] + 0.001, 3)

    meta = {
        "aligner": "wav2vec2-whisperx",
        "align_model": _ALIGN_MODEL or "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        "ref_words": len(ref),
        "aligned_units": matched,
        "coverage": round(matched / len(ref), 3) if ref else 0.0,
        "interpolated": sum(interp_flags),
        "snapped_to_silence": snapped,
        "wt": len(word_timeline),
        "ct": len(char_timeline),
        "device": device,
    }
    return {"meta": meta, "timeline": timeline,
            "word_timeline": word_timeline, "char_timeline": char_timeline}


def _interp_missing(bounds):
    """Дыры (ref-слова без пары) — линейной интерполяцией между известными соседями.

    Интерполированным конца не даём (нулевая длина) — реального конца у них нет.
    Возвращает (bounds_full, interp_flags).
    """
    n = len(bounds)
    known = [i for i, b in enumerate(bounds) if b is not None]
    out = [(0.0, 0.0)] * n
    flags = [False] * n
    if not known:
        return out, flags
    for i in range(n):
        if bounds[i] is not None:
            out[i] = bounds[i]; continue
        flags[i] = True
        left = max([k for k in known if k < i], default=None)
        right = min([k for k in known if k > i], default=None)
        if left is not None and right is not None:
            t0 = bounds[left][1]; t1 = bounds[right][0]
            frac = (i - left) / (right - left)
            t = t0 + (t1 - t0) * frac
            out[i] = (t, t)
        elif left is not None:
            out[i] = (bounds[left][1], bounds[left][1])
        else:
            out[i] = (bounds[right][0], bounds[right][0])
    return out, flags
