"""Возвраты/перечитки чтеца (П8) из СВОЕЙ акустики w2v — БЕЗ данных forced/MMS (директива 24.07).

wav2vec2 «слышит» распевный таджвид там, где MMS слепа (0% букв на мелодике) → детект возвратов на
w2v-эмиссиях покрывает и мелодичные перечитки, которые forced/`_inherit_repeats` пропускали
(rec7 6:101 «وخلق كل شيء ×2»). Работает поверх готового word_timeline (тайминги слов) + эмиссий.

Два случая в ДЫРЕ (большой разрыв между концом слова i и началом i+1, где есть РЕЧЬ):
  1. Форвард-филл (мелодичная перечитка ВПЕРЁДНЕЙ фразы): декод дыры матчит ВПЕРЁДНИЕ слова [i+1..j]
     сильно лучше, чем ТЯНУТОЕ предыдущее слово i (мадд). → чтец читает вперёд (в дыре — первая,
     распевная копия; чёткая копия уже выровнена ПОСЛЕ дыры) → заполняем дыру словами [i+1..j] →
     подсветка едет сквозь распев, а не залипает на i; вместе с пост-дырной копией даёт честный ×2.
     Дискриминатор ОТНОСИТЕЛЬНЫЙ (fwd >> held), не абсолютный порог — робастен к шумному декоду
     (rec7: fwd 0.36–0.41 vs held 0.05–0.16, разделение 2.6–6×).
  2. Назадний возврат (классический П8, خضرا): декод дыры матчит слова НАЗАД [ra..i] лучше вперёда
     → чтец вернулся и перечитывает → вклейка «назадних» точек (как falign._detect_repeats, но на
     арабском скелете w2v).

Fail-safe: любая ошибка → нет вставок.
"""
from __future__ import annotations

import os

import falign  # переиспуем _frame_db/_lev/_sim/_collapse_tandem (общие кирпичи, не данные источника)

SAMPLE_RATE = 16000

_MIN_GAP = 1.0            # с: короче — не разрыв (внутрисловные паузы таджвида)
_MIN_SPEECH = 0.35        # доля кадров-речи в повторной части дыры
_MIN_DECODE = 4           # символов скелета в декоде (меньше — шум)
_LOOKBACK = 12            # на сколько слов назад искать возврат
_FWD_SPAN = 8             # на сколько слов вперёд пробовать форвард-фразу
_MIN_SIM = 0.30           # мин. абсолютная похожесть скелетов (декод шумный → ниже, чем у MMS 0.5)
_FWD_MARGIN = 0.10        # fwd должно обгонять held/back хотя бы на столько
_PAUSE_MIN = 0.15         # мин. пауза, чтобы якорить повтор на её конец
_SNAP_FRAME_MS = 20

# согласный скелет арабского: долой харакаты + слабые/долгие гласные + носители хамзы
_DROP = set("ًٌٍَُِّْٰٕٖٓٔٗ٘") | set("اآأإٱٲٳءئؤوىة")


def _askel(word: str) -> str:
    return "".join(c for c in word if c not in _DROP)


def detect(emissions, stride_ms, idx2ch, ch2idx, word_timeline, verses, audio_path):
    """Вернуть список word_timeline-точек перечиток (rep=True) для вклейки. При проблеме → []."""
    try:
        import numpy as np
        import quran as quranmod
        import w2v_align

        special = {ch2idx.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>", "|", "-", "ـ")} - {None}
        blank = ch2idx.get("<pad>", 0)

        # ref в порядке чтения + скелеты
        ref = []
        for s, a, txt in verses:
            for wi, w in enumerate(quranmod.word_tokens(txt)):
                ref.append((s, a, wi, w))
        n = len(ref)
        if n < 2:
            return []
        skel = [_askel(w) for (_, _, _, w) in ref]

        # bounds по word_timeline (первое вхождение каждого (surah,ayah,wi))
        bt = {}
        for e in word_timeline:
            key = (e["surah"], e["ayah"], e["wi"])
            if key not in bt:
                bt[key] = (e["t"], e.get("t_end", e["t"]))
        bounds = [bt.get((s, a, wi)) for (s, a, wi, _) in ref]

        # greedy-CTC декод окна эмиссий → арабский согласный скелет
        def decode(t0, t1):
            f0 = max(0, int(t0 * 1000 / stride_ms)); f1 = min(emissions.shape[0], int(t1 * 1000 / stride_ms))
            if f1 <= f0:
                return ""
            ids = emissions[f0:f1].argmax(axis=1)
            out, prev = [], -1
            for a in ids:
                a = int(a)
                if a != prev and a not in special:
                    out.append(idx2ch.get(a, ""))
                prev = a
            return _askel("".join(out))

        # RMS-речь (аудио грузим тем же независимым загрузчиком, что и forced_align — без whisperx)
        audio = w2v_align._load_wav(audio_path)
        frame_len = max(1, int(SAMPLE_RATE * _SNAP_FRAME_MS / 1000))
        db = falign._frame_db(audio, frame_len)
        speech = None
        if db is not None and len(db) >= 3:
            thr = float(np.percentile(db, 15)) + 10.0
            speech = db >= thr
        frame_sec = frame_len / SAMPLE_RATE

        def speech_frac(t0, t1):
            if speech is None:
                return 1.0
            a, b = int(t0 / frame_sec), int(t1 / frame_sec)
            a, b = max(0, a), min(len(speech), b)
            return float(speech[a:b].mean()) if b > a else 0.0

        def repeat_onset(t0, t1):
            """Начало повтора = конец самой длинной паузы в [t0,t1] (до неё держим слово-остановку)."""
            if speech is None:
                return t0
            a, b = max(0, int(t0 / frame_sec)), min(len(speech), int(t1 / frame_sec))
            if b - a < 2:
                return t0
            seg = speech[a:b]; best_len, best_end, run = 0, a, 0
            for j in range(len(seg)):
                if not seg[j]:
                    run += 1
                else:
                    if run > best_len:
                        best_len, best_end = run, a + j
                    run = 0
            return best_end * frame_sec if best_len * frame_sec >= _PAUSE_MIN else t0

        inserts = []
        for i in range(n - 1):
            if not bounds[i] or not bounds[i + 1]:
                continue
            g0, g1 = bounds[i][1], bounds[i + 1][0]
            if g1 - g0 < _MIN_GAP:
                continue
            onset = repeat_onset(g0, g1)
            if speech_frac(onset, g1) < _MIN_SPEECH:
                continue
            dec = falign._collapse_tandem(decode(onset, g1))
            if len(dec) < _MIN_DECODE:
                continue

            # форвард: [i+1..j] × reps
            fbest = (0.0, None)  # (sim, (reps, j))
            for reps in (1, 2, 3):
                cand = ""
                for j in range(i + 1, min(n, i + 1 + _FWD_SPAN)):
                    cand += skel[j]
                    if not cand:
                        continue
                    sm = falign._sim(dec, cand * reps)
                    if sm > fbest[0]:
                        fbest = (sm, (reps, j))
            # тянутое предыдущее слово i (мадд)
            held = max((falign._sim(dec, skel[i] * k) for k in range(1, 15)), default=0.0) if skel[i] else 0.0
            # назад: [ra..i]
            bbest = (0.0, None)  # (sim, ra)
            for ra in range(max(0, i - _LOOKBACK), i):
                cand = "".join(skel[ra:i + 1])
                if cand:
                    sm = falign._sim(dec, cand)
                    if sm > bbest[0]:
                        bbest = (sm, ra)

            span = g1 - onset
            if fbest[0] >= _MIN_SIM and fbest[0] - held >= _FWD_MARGIN and fbest[0] >= bbest[0]:
                # форвард-филл: слова [i+1..j] в дыру (первая, распевная копия)
                _reps, j = fbest[1]
                rng = list(range(i + 1, j + 1))
                _distribute(inserts, ref, skel, rng, onset, span)
            elif bbest[0] >= _MIN_SIM and bbest[0] - fbest[0] >= _FWD_MARGIN and bbest[0] - held >= _FWD_MARGIN:
                # назадний возврат: [ra..i]
                ra = bbest[1]
                rng = list(range(ra, i + 1))
                _distribute(inserts, ref, skel, rng, onset, span)
        return inserts
    except Exception:
        return []


def _distribute(inserts, ref, skel, rng, onset, span):
    """Раздать время [onset, onset+span] словам rng пропорц. длине их скелета → точки rep=True."""
    lens = [max(1, len(skel[k])) for k in rng]
    total = sum(lens)
    acc = onset
    for off, k in enumerate(rng):
        t0 = acc
        acc += span * lens[off] / total
        su, ay, wi, _ = ref[k]
        inserts.append({"t": round(t0, 3), "t_end": round(acc, 3),
                        "surah": su, "ayah": ay, "wi": wi, "rep": True})
