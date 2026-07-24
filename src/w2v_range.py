"""Независимое определение диапазона аятов из wav2vec2-эмиссий — БЕЗ ASR (google/whisper/forced).

w2v — самодостаточный источник (директива владельца 24.07): audio → wav2vec2 CTC-эмиссии → САМ
находит, что читается. Greedy-декод мелодичного таджвида беден (модель молчит на распеве, ~84%
blank), НО буквы, что есть, ложатся на текст → чисто БУКВЕННЫЙ поиск диапазона (подход владельца,
difflib), без CTC/GPU.

Диапазон = ПРОИЗВОЛЬНЫЙ непрерывный отрезок аятов (указка владельца 24.07: чтение бывает частью
суры и может пересекать границы сур — не привязываемся к «одна сура»/«вся сура»). По ПЛОСКОМУ
индексу аятов всего Корана:
  1. Локализация (дёшево): greedy-декод → согласный скелет → k-граммы → плотность попаданий по
     плоским аятам → densest непрерывный кластер аятов (± запас; отсекает 90%+ Корана).
  2. Точно: difflib-скоринг (подход владельца — и совпадения, и промежутки несовпадения:
     SequenceMatcher.ratio) окон вокруг кластера, ПОЛНЫЙ перебор → максимум ratio = истинное окно.
     Проверено: rec7 6:95-103, rec5 25:63-77 — ровно истина, <1с.

Всё на CPU, БЕЗ GPU. Эмиссии (для greedy-декода) считает `w2v_align.emissions()` (GPU).
Резерв: CTC-forward-скоринг (`ctc_logprob`) — дороже, но тоже без GPU и тоже даёт истинное окно.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from quran import _FOLD_TABLE, _STRIP_TABLE

_K = 5                    # длина k-граммы для буквенной локализации
_NEG = -1e9
_REGION_MARGIN = 6        # ± аятов запаса вокруг плотного кластера (CTC добьёт точную границу)


# --- greedy-декод эмиссий → согласный скелет (нормализация как у корпуса quran) ---

def greedy_skeleton(emissions: np.ndarray, idx2ch: dict, special: set, times: bool = False,
                    stride_ms: float = 20.0):
    """argmax + collapse + выброс blank/служебных → буквы, нормализованные как корпус (fold+strip).
    times=True → вернуть (skel, char_times) где char_times[k] — время (с) появления символа k."""
    ids = emissions.argmax(axis=1)
    out, ts, prev = [], [], -1
    for fi, a in enumerate(ids):
        a = int(a)
        if a != prev and a not in special:
            ch = idx2ch.get(a, "").translate(_STRIP_TABLE).translate(_FOLD_TABLE)
            for c in ch:
                out.append(c); ts.append(fi * stride_ms / 1000.0)
        prev = a
    return ("".join(out), ts) if times else "".join(out)


def ayah_start_hints(emissions, verses, index, idx2ch, ch2idx, stride_ms):
    """Старты аятов (с) из СВОЕЙ акустики — для нарезки длинного аудио в force-align БЕЗ ASR.

    Выравниваем ВЕСЬ decode-скелет к склеенному согласному тексту диапазона одним difflib →
    matching-блоки дают соответствие decode-позиция↔позиция-в-тексте. Для КАЖДОЙ границы аята
    (накопленный char-offset в тексте) берём ближайший блок → decode-позиция → время (по stride).
    Плотнее и надёжнее k-грамм (у мелодичных аятов k-грамм-попаданий нет, а difflib-блок находится
    от соседей). verses=[(surah,ayah,text)]. Возвращает старты (с) по verses (None где не легло)."""
    import difflib
    special = {ch2idx.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>", "|", "-", "ـ")} - {None}
    _Cs, _char2fa, _kidx, _flat, _fa_skel = index
    dec, ts = greedy_skeleton(emissions, idx2ch, special, times=True, stride_ms=stride_ms)

    from quran import normalize
    ay_skel = [normalize(t).replace(" ", "") for _, _, t in verses]
    ref = "".join(ay_skel)
    # накопленный char-offset начала каждого аята в ref
    offsets, acc = [], 0
    for sk in ay_skel:
        offsets.append(acc); acc += len(sk)

    sm = difflib.SequenceMatcher(None, dec, ref, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]   # (a=decode_pos, b=ref_pos, size)
    if not blocks:
        return [None] * len(verses)

    def ref_to_time(off):
        """Ближайшая по ref-позиции точка соответствия → её decode-время."""
        best = None
        for b in blocks:
            if b.b <= off < b.b + b.size:            # offset внутри блока → точное соответствие
                return ts[b.a + (off - b.b)]
            d = min(abs(b.b - off), abs(b.b + b.size - off))
            if best is None or d < best[0]:
                best = (d, b.a)
        return ts[best[1]] if best else None

    starts = [ref_to_time(off) for off in offsets]
    # монотонизируем (старты аятов растут; выбросы назад/невалидные → None → интерполяция _fill_starts)
    mono, last = [], -1.0
    for x in starts:
        if x is None or x < last:
            mono.append(None)
        else:
            mono.append(x); last = x
    return mono


def build_index(quran):
    """Один раз: карты char→(плоский аят), инвертированный k-грамм-индекс, плоский список аятов
    (surah,ayah) в порядке корпуса + согласный скелет текста каждого плоского аята (для difflib)."""
    flat_ayahs = []                 # [(surah, ayah)] уникально, в порядке корпуса
    fa_text = []                    # нормализованный (безхаракатный) текст каждого плоского аята
    C, char2fa = [], []             # char2fa[pos] = индекс в flat_ayahs
    last = None
    for t in quran.tokens:
        key = (t.surah, t.ayah)
        if key != last:
            flat_ayahs.append(key); fa_text.append([]); last = key
        fa = len(flat_ayahs) - 1
        fa_text[fa].append(t.text)
        for ch in t.text:
            C.append(ch); char2fa.append(fa)
    Cs = "".join(C)
    fa_skel = ["".join(words) for words in fa_text]   # t.text уже нормализован (без харакат) в корпусе
    kidx = defaultdict(list)
    for p in range(len(Cs) - _K + 1):
        kidx[Cs[p:p + _K]].append(p)
    return Cs, char2fa, kidx, flat_ayahs, fa_skel


# --- буквенная локализация: плотный кластер плоских аятов ---

def _ayah_density(skel: str, char2fa, kidx, n_fa: int) -> np.ndarray:
    """Число k-грамм-попаданий декода на каждый плоский аят (буквенно, дёшево)."""
    dens = np.zeros(n_fa, dtype=np.int32)
    for sp in range(len(skel) - _K + 1):
        for cp in kidx.get(skel[sp:sp + _K], ()):
            dens[char2fa[cp]] += 1
    return dens


def _dense_region(dens: np.ndarray) -> tuple[int, int] | None:
    """Densest непрерывный кластер плоских аятов. Сглаживаем плотность, берём пик, расширяем пока
    плотность выше фона. Отсекает рассеянный шум (ложные k-граммы по всему Корану)."""
    if dens.sum() == 0:
        return None
    n = len(dens)
    # сглаживание окном 3 (аяты рядом с читаемыми тоже ловят попадания)
    k = np.array([1.0, 1.0, 1.0])
    sm = np.convolve(dens.astype(float), k, mode="same")
    peak = int(sm.argmax())
    floor = float(np.percentile(sm[sm > 0], 50)) if (sm > 0).any() else 0.0
    thr = max(1.0, floor)
    lo = hi = peak
    while lo - 1 >= 0 and sm[lo - 1] >= thr:
        lo -= 1
    while hi + 1 < n and sm[hi + 1] >= thr:
        hi += 1
    return lo, hi


# --- CTC-forward скоринг (лог-пространство, векторизовано по S) ---

def pool_emissions(emis: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool лог-вероятностей по времени (factor кадров → 1). Абсолютный score меняется, но
    ОТНОСИТЕЛЬНЫЙ ранкинг окон сохраняется (до factor≈4; при ≥8 CTC ломается на коротком T) →
    argmax окно то же, а CTC-forward в factor раз быстрее."""
    if factor <= 1:
        return emis
    T, V = emis.shape
    n = T // factor
    if n == 0:
        return emis
    return emis[:n * factor].reshape(n, factor, V).mean(axis=1)


def _text_to_ids(text: str, ch2idx: dict, blank: int) -> list[int]:
    """Текст аята → id-последовательность vocab модели (буквы + ХАРАКАТЫ, что есть в vocab).

    ⚠️ Владелец (24.07) предлагал скорить без харакатов (их плохо распознают). Для СЛОВЕСНОГО ПОИСКА
    (k-граммы) так и есть — там нормализованный (безхаракатный) текст. Но для CTC-скоринга ЭМПИРИКА
    обратная: с харакатами истинное окно выигрывает (rec7 −3.8 vs неверные −4.4), БЕЗ них ранкинг
    ломается. Причина: wav2vec2 ЧАСТЬ харакатов эмитит (фатха/кясра/сукун), они дают дискриминацию;
    недоэмиченные CTC-forward разруливает сам через blank. Поэтому харакаты (что в vocab) ОСТАВЛЯЕМ."""
    ids = []
    for ch in text:
        if ch == " ":
            continue
        j = ch2idx.get(ch)
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


def _difflib_score(dec: str, ref: str) -> float:
    """Качество выравнивания декода к тексту отрезка (подход владельца): и совпадения, и промежутки
    несовпадения. difflib.ratio() = 2·matched/(len(dec)+len(ref)) — учитывает и матчи, и «дыры»
    (несматченные куски штрафуют знаменателем). Максимум ratio по окнам = истинный диапазон
    (проверено: rec7 6:95-103, rec5 25:63-77 — ровно истина). Быстро (C), без CTC/GPU."""
    if not ref or not dec:
        return 0.0
    import difflib
    return difflib.SequenceMatcher(None, dec, ref, autojunk=False).ratio()


def find_range(emissions: np.ndarray, quran, idx2ch: dict, ch2idx: dict,
               index=None, verbose: bool = False) -> list[tuple[int, int]] | None:
    """Главный вход: список (surah, ayah) читаемого диапазона из эмиссий (по порядку). None если нет.

    Диапазон — произвольный непрерывный отрезок плоских аятов (часть суры / через границу сур —
    указка владельца). Чисто буквенно (без CTC/GPU): (1) k-грамм-плотность → плотный кластер аятов;
    (2) difflib-добор границ (совпадения+промежутки) → максимум ratio = истинное окно."""
    special = {ch2idx.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>", "|", "-", "ـ")} - {None}
    Cs, char2fa, kidx, flat_ayahs, fa_skel = index or build_index(quran)
    n_fa = len(flat_ayahs)

    dec = greedy_skeleton(emissions, idx2ch, special)
    if len(dec) < _K:
        return None

    # 1) буквенная локализация → плотный кластер плоских аятов (± запас)
    dens = _ayah_density(dec, char2fa, kidx, n_fa)
    reg = _dense_region(dens)
    if reg is None:
        return None
    lo = max(0, reg[0] - _REGION_MARGIN)
    hi = min(n_fa - 1, reg[1] + _REGION_MARGIN)
    if verbose:
        s0, a0 = flat_ayahs[lo]; s1, a1 = flat_ayahs[hi]
        print(f"буквенный регион: {s0}:{a0}..{s1}:{a1} ({hi-lo+1} аятов)")

    # 2) ПОЛНЫЙ перебор окон в регионе (difflib дёшев): для каждого старта i0 — все концы i1 в
    # пределах разумной длины; берём максимум ratio. Жадный добор застревал в локальном оптимуме
    # (rec7 6:95-110 ratio 0.313 вместо истинного 6:95-103 0.338); перебор находит глобальный.
    max_len = hi - lo + 1
    best = (-1.0, lo, lo)
    for i0 in range(lo, hi + 1):
        for i1 in range(i0, min(hi, i0 + max_len) + 1):
            v = _difflib_score(dec, "".join(fa_skel[i0:i1 + 1]))
            if v > best[0]:
                best = (v, i0, i1)
    cur, i0, i1 = best
    verses = flat_ayahs[i0:i1 + 1]
    if verbose:
        s0, a0 = verses[0]; s1, a1 = verses[-1]
        print(f"диапазон: {s0}:{a0}..{s1}:{a1}  difflib-ratio={cur:.3f}")
    return verses
