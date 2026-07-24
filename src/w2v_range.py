"""Независимое определение диапазона аятов из wav2vec2-эмиссий — БЕЗ ASR (google/whisper/forced).

w2v — самодостаточный источник (директива владельца 24.07): audio → wav2vec2 CTC-эмиссии → САМ
находит, что читается. Greedy-декод мелодичного таджвида беден (модель молчит на распеве, ~84%
blank) → словесный/char-матчинг ненадёжен для точного диапазона. Робастный сигнал — CTC-forward-
скоринг: logP(эмиссии | текст-кандидат) правильно моделирует blank'и. Истинное окно = максимум
нормированного score (доказано: rec7 6:95-103 norm/char −3.803 против неверных −4.4..−5.9).

Двухступенчато (CTC-forward на весь корпус дорог):
  1. Грубо: greedy-декод → согласный скелет → k-граммы → топ-N сур-кандидатов (истинная сура
     стабильно в топе; точность не нужна, нужен recall).
  2. Точно: внутри кандидат-сур скользящее окно → CTC-скоринг → лучший центр → добор/сжатие до
     границ аятов по максимуму score. Возвращает (surah, ayah_lo, ayah_hi).

CTC-скоринг — numpy, БЕЗ GPU (гоняется на сохранённых/переданных эмиссиях). Эмиссии считает
`w2v_align.emissions()` (GPU).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from quran import _FOLD_TABLE, _STRIP_TABLE

_K = 5                    # длина k-граммы для грубого префильтра
_TOPN_SURAHS = 8          # сколько сур-кандидатов брать в CTC-поиск (recall > precision)
_NEG = -1e9


# --- greedy-декод эмиссий → согласный скелет (нормализация как у корпуса quran) ---

def greedy_skeleton(emissions: np.ndarray, idx2ch: dict, special: set) -> str:
    """argmax + collapse + выброс blank/служебных → буквы, нормализованные как корпус (fold+strip)."""
    ids = emissions.argmax(axis=1)
    out, prev = [], -1
    for a in ids:
        a = int(a)
        if a != prev and a not in special:
            out.append(idx2ch.get(a, "").translate(_STRIP_TABLE).translate(_FOLD_TABLE))
        prev = a
    return "".join(out)


def _corpus_chars(quran):
    """Плоский char-поток нормализованного корпуса + карта позиция→token_index."""
    C, char2tok = [], []
    for ti, t in enumerate(quran.tokens):
        for ch in t.text:
            C.append(ch); char2tok.append(ti)
    return "".join(C), char2tok


def _kmer_hits(skel: str, kidx):
    """Все попадания k-грамм декода в корпус: список corpus-позиций cp."""
    cps = []
    for sp in range(len(skel) - _K + 1):
        cps.extend(kidx.get(skel[sp:sp + _K], ()))
    return cps


def candidate_surahs(cps, quran, char2tok, surlen, topn: int = _TOPN_SURAHS) -> list[int]:
    """Топ-N сур по числу k-грамм-попаданий декода, нормированному на sqrt(длину суры).
    Recall-ориентировано: истинная сура должна ПОПАСТЬ в список (не обязательно первой)."""
    raw = Counter(quran.tokens[char2tok[cp]].surah for cp in cps)
    if not raw:
        return list(range(1, min(topn, 114) + 1))
    scored = sorted(raw, key=lambda s: raw[s] / (surlen[s] ** 0.5), reverse=True)
    return scored[:topn]


def _dense_region(cps, quran, char2tok, surah: int, margin: int = 8) -> tuple[int, int] | None:
    """Плотный регион аятов суры по k-грамм-попаданиям (буквенная локализация — дёшево).
    Отсекает одиночные ложные попадания (порог по плотности), берёт мин..макс плотных аятов + запас.
    CTC потом сканит ТОЛЬКО этот регион, не всю суру."""
    ays = [quran.tokens[char2tok[cp]].ayah for cp in cps if quran.tokens[char2tok[cp]].surah == surah]
    if not ays:
        return None
    ayc = Counter(ays)
    thr = max(2, len(ays) // 30)
    dense = sorted(a for a, n in ayc.items() if n >= thr)
    if not dense:
        dense = sorted(ayc)
    na = quran.surah(surah).verses_count
    return max(1, dense[0] - margin), min(na, dense[-1] + margin)


def build_index(quran):
    """Один раз: char-поток корпуса, карта, инвертированный k-грамм-индекс, длины сур."""
    Cs, char2tok = _corpus_chars(quran)
    kidx = defaultdict(list)
    for p in range(len(Cs) - _K + 1):
        kidx[Cs[p:p + _K]].append(p)
    surlen = Counter(t.surah for t in quran.tokens)
    return Cs, char2tok, kidx, surlen


# --- CTC-forward скоринг (лог-пространство, векторизовано по S) ---

def _text_to_ids(text: str, ch2idx: dict, blank: int) -> list[int]:
    """Текст аята → id-последовательность vocab модели (буквы + ХАРАКАТЫ, что есть в vocab).

    ⚠️ Владелец (24.07) предлагал скорить без харакатов (их плохо распознают). Для СЛОВЕСНОГО ПОИСКА
    (difflib/k-граммы) так и есть — там нормализованный (безхаракатный) текст. Но для CTC-скоринга
    ЭМПИРИКА обратная: с харакатами истинное окно выигрывает (rec7 6:95-103 −3.8 vs неверные −4.4),
    БЕЗ харакатов ранкинг ломается (6:90-103/7:1-23 обгоняют истину). Причина: wav2vec2 ЧАСТЬ
    харакатов эмитит (фатха ~3%, кясра, сукун), они добавляют дискриминации; недоэмиченные CTC-forward
    разруливает сам через blank. Поэтому в CTC-таргете харакаты ОСТАВЛЯЕМ (только те, что в vocab)."""
    ids = []
    for ch in text:
        if ch == " ":
            continue
        j = ch2idx.get(ch)          # символы не из vocab (дагер-алиф, вакфы) отсеются сами
        if j is not None and j != blank:
            ids.append(j)
    return ids


def ctc_logprob(emis: np.ndarray, labels: list[int], blank: int) -> float:
    """log P(labels | emis) CTC-forward'ом (лог-пространство). Больше = лучше. Пустые → _NEG."""
    T = emis.shape[0]
    if not labels or T == 0:
        return _NEG
    ext = np.array([blank] + [x for l in labels for x in (l, blank)], dtype=np.int64)
    S = len(ext)
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    emis_ext = emis[:, ext]                          # (T,S)
    a = np.full(S, _NEG, dtype=np.float64)
    a[0] = emis_ext[0, 0]
    if S > 1:
        a[1] = emis_ext[0, 1]
    for t in range(1, T):
        a1 = np.empty(S); a1[0] = _NEG; a1[1:] = a[:-1]
        a2 = np.full(S, _NEG); a2[2:] = np.where(skip[2:], a[:-2], _NEG)
        a = np.logaddexp(np.logaddexp(a, a1), a2) + emis_ext[t]
    return float(np.logaddexp(a[S - 1], a[S - 2]) if S > 1 else a[0])


def pool_emissions(emis: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool лог-вероятностей по времени (factor кадров → 1). Абсолютный score меняется, но
    ОТНОСИТЕЛЬНЫЙ ранкинг окон сохраняется (все скорятся на одних пулингованных эмиссиях) → argmax
    окно то же, а CTC-forward в factor раз быстрее (T↓). Для грубого скана диапазона этого хватает."""
    if factor <= 1:
        return emis
    T, V = emis.shape
    n = T // factor
    if n == 0:
        return emis
    return emis[:n * factor].reshape(n, factor, V).mean(axis=1)


def _ayah_text(quran, surah: int, ayah: int) -> str:
    try:
        return quran.surah(surah).verses[ayah - 1].text
    except Exception:
        return ""


def _score_span(emis, quran, surah, a0, a1, ch2idx, blank) -> tuple[float, int]:
    """Нормированный на длину меток logP окна [a0..a1] суры. Возвращает (norm_score, n_labels)."""
    txt = " ".join(_ayah_text(quran, surah, a) for a in range(a0, a1 + 1))
    ids = _text_to_ids(txt, ch2idx, blank)
    if not ids:
        return _NEG, 0
    return ctc_logprob(emis, ids, blank) / len(ids), len(ids)


def find_range(emissions: np.ndarray, quran, idx2ch: dict, ch2idx: dict,
               index=None, pool: int = 2, verbose: bool = False) -> tuple[int, int, int] | None:
    """Главный вход: (surah, ayah_lo, ayah_hi) читаемого диапазона из эмиссий. None если не найдено.

    pool — фактор mean-pool эмиссий для скоринга (скорость; ОТНОСИТЕЛЬНЫЙ ранкинг окон сохраняется
    до pool≈4, при pool≥8 CTC ломается на коротком T). Весь поиск/добор — на одних pooled-эмиссиях,
    поэтому сравнимо. Точные тайминги не тут — их даёт последующий force-align диапазона."""
    blank = ch2idx.get("<pad>", 0)
    special = {ch2idx.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>", "|", "-", "ـ")} - {None}
    Cs, char2tok, kidx, surlen = index or build_index(quran)

    skel = greedy_skeleton(emissions, idx2ch, special)
    if len(skel) < _K:
        return None
    cps = _kmer_hits(skel, kidx)
    cand = candidate_surahs(cps, quran, char2tok, surlen)
    if verbose:
        print("кандидат-суры:", cand)

    emis = pool_emissions(emissions, pool)

    def score(s, a0, a1):
        return _score_span(emis, quran, s, a0, a1, ch2idx, blank)[0]

    # В КАЖДОЙ суре-кандидате: буквенный регион (k-граммы) сужает поиск → грубое окно внутри региона
    # → жадный добор границ по максимуму нормированного score. Сравниваем ДОБОРАННЫЕ score всех
    # кандидатов (тесное истинное окно обгоняет спурьёзные; rec7 6:95-103). Все на pooled-эмиссиях.
    refined = []
    for s in cand:
        na = quran.surah(s).verses_count
        region = _dense_region(cps, quran, char2tok, s) or (1, na)
        r_lo, r_hi = region
        W = min(r_hi - r_lo + 1, 12)
        step = max(1, W // 3)
        sbest = None
        a0 = r_lo
        while a0 <= r_hi:
            a1 = min(r_hi, a0 + W - 1)
            sc0 = score(s, a0, a1)
            if sbest is None or sc0 > sbest[0]:
                sbest = (sc0, a0, a1)
            if a1 >= r_hi:
                break
            a0 += step
        cur, a0, a1 = sbest
        improved = True
        while improved:
            improved = False
            for na0, na1 in ((a0 - 1, a1), (a0 + 1, a1), (a0, a1 + 1), (a0, a1 - 1),
                             (a0 + 1, a1 + 1), (a0 - 1, a1 - 1)):
                if na0 < 1 or na1 > na or na0 > na1:
                    continue
                v = score(s, na0, na1)
                if v > cur + 1e-6:
                    cur, a0, a1 = v, na0, na1
                    improved = True
                    break
        refined.append((cur, s, a0, a1))
        if verbose:
            print(f"  кандидат {s}:{a0}..{a1}  norm/char={cur:.3f}")
    if not refined:
        return None
    refined.sort(key=lambda x: -x[0])
    cur, s, a0, a1 = refined[0]
    if verbose:
        print(f"диапазон: {s}:{a0}..{s}:{a1}  norm/char={cur:.3f}")
    return s, a0, a1
