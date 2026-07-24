"""Возвраты/перечитки чтеца (П8) из СВОЕЙ акустики w2v — БЕЗ данных forced/MMS (директива 24.07).

wav2vec2 «слышит» распевный таджвид там, где MMS слепа → детект возвратов на w2v-эмиссиях покрывает
и мелодичные перечитки, которые forced/`_inherit_repeats` пропускали (rec7 6:101 صاحبة وخلق كل شيء).

ПОЧЕМУ НЕ ПО ДЫРАМ. Монотонный CTC-Viterbi (w2v_align) по построению НЕ оставляет дыр — слово владеет
временем до онсета следующего. Поэтому старый gap-детектор (как в falign) на w2v молчал (reps=0).
Возврат в w2v выглядит иначе: span слова-остановки РАСТЯГИВАЕТСЯ и поглощает перечитку/распев. Значит
триггер — не дыра, а АНОМАЛЬНО ДЛИННЫЙ span, а внутри него — акустический разбор:

  1. FWD (мелодичный обгон / перечитка впередней фразы): декод span сильно похож на СЛЕДУЮЩИЕ слова
     [i+1..j], а НЕ на растянутый держатель i (мадд). Чтец поёт i, i+1, … — подсветка иначе залипает на
     i. → раздаём [i+1..j] по span (первая, распевная копия); чёткая копия/продолжение уже выровнены.
     rec7 6:101: держатель صاحبة, декод ≈ خلق كل شيء → подсветка едет صاحبة→وخلق→كل→شيء.
  2. BACK (классический возврат, خضرا): декод span похож на слова НАЗАД [ra..i] лучше вперёда/мадда →
     чтец вернулся и перечитывает → «назадние» точки в хвост span (после паузы-остановки).

Дискриминаторы (робастность к шумному декоду распева): (а) span заметно длиннее медианного;
(б) декод многословный (len ≥ 2× скелет держателя) — иначе это чистый мадд; (в) победившее направление
обгоняет и held-нуль (держатель как мадд), и другое направление; (г) сбалансированный difflib-ratio
(штрафует расхождение длин cand↔dec) — не одностороннее покрытие (короткий кандидат покрывается шумом).
Пропускаем интро (первые слова — истиаза/басмала) и односимвольные держатели (ненадёжны).

Fail-safe: любая ошибка → нет вставок.
"""
from __future__ import annotations

import os

import falign  # переиспуем _frame_db/_sim/_collapse_tandem (общие кирпичи, не данные источника)

SAMPLE_RATE = 16000

_MIN_SPAN_ABS = 2.0       # с: короче — не кандидат-держатель
_SPAN_FACTOR = 2.5        # и span ≥ столько медиан
_MIN_DECODE = 8           # символов скелета в декоде span (меньше — шум/короткий мадд)
_DEC_LEN_FACTOR = 2.0     # декод ≥ столько× скелета держателя (многословное содержание, не мадд)
_MIN_SIM = 0.42           # мин. похожесть победившего направления (декод шумный → ниже, чем у MMS 0.5)
_HELD_MARGIN = 0.12       # победитель обгоняет held-нуль (держатель-мадд) хотя бы на столько
_DIR_MARGIN = 0.08        # и обгоняет другое направление хотя бы на столько
_MIN_HOLDER_SKEL = 2      # односимвольные держатели (напр. إِلَّا→'ل') ненадёжны
_INTRO_SKIP = 3           # первые N слов диапазона — интро (истиаза/басмала), не держатели
_LOOKBACK = 12            # на сколько слов назад искать возврат
_FWD_SPAN = 8             # на сколько слов вперёд пробовать форвард-фразу
_PAUSE_MIN = 0.15         # мин. пауза, чтобы якорить назадний повтор на её конец
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

        special = {ch2idx.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>", "|", "-", "ـ")} - {None}

        # ref в порядке чтения + скелеты
        ref = []
        for s, a, txt in verses:
            for wi, w in enumerate(quranmod.word_tokens(txt)):
                ref.append((s, a, wi, w))
        n = len(ref)
        if n < 2:
            return []
        skel = [_askel(w) for (_, _, _, w) in ref]

        # span каждого ref-слова из word_timeline (первое вхождение (surah,ayah,wi))
        bt = {}
        order = {}
        for e in word_timeline:
            key = (e["surah"], e["ayah"], e["wi"])
            if key not in bt:
                bt[key] = (e["t"], e.get("t_end"))
                order[key] = e["t"]
        onset = [bt.get((s, a, wi), (None, None))[0] for (s, a, wi, _) in ref]
        tend = [bt.get((s, a, wi), (None, None))[1] for (s, a, wi, _) in ref]
        spans = []
        for i in range(n):
            t0 = onset[i]
            if t0 is None:
                spans.append(None); continue
            t1 = tend[i]
            if t1 is None:
                t1 = onset[i + 1] if i + 1 < n and onset[i + 1] is not None else t0
            spans.append((t0, t1))
        dur = [b - a for s2 in spans if s2 for a, b in (s2,) if b > a]
        med = float(np.median(dur)) if dur else 1.0
        span_thr = max(_MIN_SPAN_ABS, _SPAN_FACTOR * med)

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
        import w2v_align
        audio = w2v_align._load_wav(audio_path)
        frame_len = max(1, int(SAMPLE_RATE * _SNAP_FRAME_MS / 1000))
        db = falign._frame_db(audio, frame_len)
        speech = None
        if db is not None and len(db) >= 3:
            thr = float(np.percentile(db, 15)) + 10.0
            speech = db >= thr
        frame_sec = frame_len / SAMPLE_RATE

        def repeat_onset(t0, t1):
            """Начало назаднего повтора = конец самой длинной паузы в [t0,t1] (до неё держим слово-стоп)."""
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
        for i in range(_INTRO_SKIP, n):
            if spans[i] is None or len(skel[i]) < _MIN_HOLDER_SKEL:
                continue
            t0, t1 = spans[i]
            if t1 - t0 < span_thr:
                continue
            dec = falign._collapse_tandem(decode(t0, t1))
            if len(dec) < _MIN_DECODE or len(dec) < _DEC_LEN_FACTOR * max(1, len(skel[i])):
                continue

            held = max((falign._sim(dec, skel[i] * k) for k in range(1, 15)), default=0.0)
            # вперёд: [i+1..j] × reps
            fbest = (0.0, None)
            for reps in (1, 2, 3):
                cand = ""
                for j in range(i + 1, min(n, i + 1 + _FWD_SPAN)):
                    cand += skel[j]
                    if cand:
                        sm = falign._sim(dec, cand * reps)
                        if sm > fbest[0]:
                            fbest = (sm, j)
            # назад: [ra..i]
            bbest = (0.0, None)
            for ra in range(max(0, i - _LOOKBACK), i):
                cand = "".join(skel[ra:i + 1])
                if cand:
                    sm = falign._sim(dec, cand)
                    if sm > bbest[0]:
                        bbest = (sm, ra)

            span = t1 - t0
            if (fbest[0] >= _MIN_SIM and fbest[0] - held >= _HELD_MARGIN
                    and fbest[0] - bbest[0] >= _DIR_MARGIN):
                # FWD: держатель i оставляет свою долю в начале span, дальше [i+1..j] едут по остатку
                j = fbest[1]
                hshare = span * len(skel[i]) / max(1, sum(len(skel[k]) for k in range(i, j + 1)))
                _distribute(inserts, ref, skel, list(range(i + 1, j + 1)), t0 + hshare, span - hshare)
            elif (bbest[0] >= _MIN_SIM and bbest[0] - held >= _HELD_MARGIN
                    and bbest[0] - fbest[0] >= _DIR_MARGIN):
                # BACK: держатель держится до конца паузы, дальше перечитка [ra..i]
                ra = bbest[1]
                onset_r = repeat_onset(t0, t1)
                _distribute(inserts, ref, skel, list(range(ra, i + 1)), onset_r, t1 - onset_r)
        return inserts
    except Exception:
        return []


def _distribute(inserts, ref, skel, rng, onset, span):
    """Раздать время [onset, onset+span] словам rng пропорц. длине их скелета → точки rep=True."""
    if not rng or span <= 0:
        return
    lens = [max(1, len(skel[k])) for k in rng]
    total = sum(lens)
    acc = onset
    for off, k in enumerate(rng):
        t0 = acc
        acc += span * lens[off] / total
        su, ay, wi, _ = ref[k]
        inserts.append({"t": round(t0, 3), "t_end": round(acc, 3),
                        "surah": su, "ayah": ay, "wi": wi, "rep": True})
