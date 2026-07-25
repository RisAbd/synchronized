"""match_align — ЕДИНЫЙ модуль матчинга распознанного к тексту Корана (директива владельца 24-25.07).

Владелец: деление «аллайнер vs распознаватель» выкинуто — источник выдаёт слова ИЛИ буквы, а
матчинг с Кораном = ОДИН переиспользуемый модуль, импортируемый каждым источником (без наследования
результатов между источниками). Здесь оба входа матчинга:

  • СЛОВЕСНЫЙ (google/whisper): seed-and-consensus по биграммам → `align(transcript, quran)` → sync_map.
    Локализация (какой аят где) и выравнивание слов идут вместе (плотный word_timeline).
  • БУКВЕННЫЙ (w2v/mms): greedy-скелет CTC-эмиссий → `find_range(emissions, quran, …)` → диапазон
    аятов (k-грамм-плотность по IDF + difflib-добор границ). Выравнивание БУКВ внутри диапазона
    делает сам источник (свой CTC-Viterbi) — источники с побуквенным выходом делят только ЛОКАЛИЗАЦИЮ.

Источники импортируют этот модуль и зовут нужный вход; собственный матчинг не пишут. Раньше словесная
часть жила в `src/align.py`, буквенная — в `src/w2v_range.py`; теперь оба — тонкие ре-экспортные шимы
над этим модулем (обратная совместимость импортов pipeline/gpu_align/run и офлайн-проб).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quran import Quran, normalize, _FOLD_TABLE, _STRIP_TABLE


# ======================= СЛОВЕСНЫЙ ВХОД (google/whisper) =======================


# --- параметры (подобраны/проверяются на реальных данных) -------------------
WINDOW = 4          # полуокно голосования за диагональ (в ASR-словах)
MIN_SUPPORT = 2     # мин. число согласных якорей, чтобы принять позицию
DIAG_TOL = 4        # допуск дрейфа диагонали внутри одного пассажа (индели/повторы)
GAP_TOL = 6         # разрыв по ASR-индексу, после которого начинается новый сегмент
MIN_SEG_WORDS = 3   # короче — считаем шумом, не сегментом
BACK_TOL = 250      # допустимый откат назад по корпусу (повторы аятов), в токенах
INTERP_MAX_GAP = 40  # макс. разрыв в словах корпуса, который добиваем интерполяцией времени


# --- контракт вход/выход ----------------------------------------------------


@dataclass
class Word:
    word: str        # исходное ASR-слово
    start: float
    end: float
    norm: str        # нормализованное


def _parse_ts(v) -> float:
    """'0.200s' | '2s' | 2.0 -> float секунд."""
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v).rstrip("s") or 0)


def load_transcript(path: str | Path) -> list[Word]:
    """Понимает формат Google STT (gstt_response.json) и наш transcript.json."""
    data = json.loads(Path(path).read_text())
    raw = []
    if isinstance(data, dict) and "results" in data:            # Google STT
        for r in data["results"]:
            for w in r["alternatives"][0]["words"]:
                raw.append((w["word"], _parse_ts(w["startTime"]), _parse_ts(w["endTime"])))
    elif isinstance(data, dict) and "words" in data:            # наш формат
        for w in data["words"]:
            raw.append((w["word"], _parse_ts(w["start"]), _parse_ts(w["end"])))
    elif isinstance(data, list):
        for w in data:
            raw.append((w["word"], _parse_ts(w["start"]), _parse_ts(w["end"])))
    else:
        raise ValueError("неизвестный формат транскрипта")

    out = []
    for word, s, e in raw:
        n = normalize(word)
        for piece in (n.split() or [""]):   # ASR-слово может нормализоваться в несколько
            out.append(Word(word=word, start=s, end=e, norm=piece))
    return [w for w in out if w.norm]


# --- индекс корпуса ---------------------------------------------------------


class CorpusIndex:
    def __init__(self, quran: Quran):
        self.q = quran
        self.words = [t.text for t in quran.tokens]
        self.by_word: dict[str, list[int]] = defaultdict(list)
        for i, w in enumerate(self.words):
            self.by_word[w].append(i)

    def bigram_positions(self, w0: str, w1: str) -> list[int]:
        """Позиции p, где corpus[p]==w0 и corpus[p+1]==w1."""
        out = []
        n = len(self.words)
        for p in self.by_word.get(w0, ()):
            if p + 1 < n and self.words[p + 1] == w1:
                out.append(p)
        return out


# --- ядро -------------------------------------------------------------------


def align(transcript: list[Word], quran: Quran, index: CorpusIndex | None = None) -> dict:
    index = index or CorpusIndex(quran)
    a = transcript
    n = len(a)

    # 1. якоря: для каждого i — список диагоналей d=c-i от биграммных совпадений
    diags_at: list[list[int]] = [[] for _ in range(n)]
    for i in range(n - 1):
        for p in index.bigram_positions(a[i].norm, a[i + 1].norm):
            diags_at[i].append(p - i)

    # 2. локальный консенсус: для каждого i голосуем по окну [i-W, i+W]
    pred: list[tuple[int, int] | None] = [None] * n   # (corpus_pos, support)
    for i in range(n):
        votes: Counter[int] = Counter()
        for j in range(max(0, i - WINDOW), min(n, i + WINDOW + 1)):
            for d in diags_at[j]:
                votes[d] += 1
        if not votes:
            continue
        d_best, support = votes.most_common(1)[0]
        if support < MIN_SUPPORT:
            continue
        c = i + d_best
        if 0 <= c < len(index.words):
            pred[i] = (c, support)

    # 3. точки sync-map
    points = []
    for i, pr in enumerate(pred):
        if pr is None:
            continue
        c, support = pr
        tok = quran.tokens[c]
        points.append({
            "t": round(a[i].start, 3),
            "t_end": round(a[i].end, 3),
            "corpus": c,
            "surah": tok.surah,
            "ayah": tok.ayah,
            "word_index": tok.word_index,
            "support": support,
            "asr_i": i,
            "asr_word": a[i].word,
        })

    # 4. сегментация: рвём при разрыве по ASR-индексу или скачке диагонали
    segments, timeline, word_timeline = _segment(points, quran)

    return {
        "meta": {
            "asr_words": n,
            "aligned_points": len(points),
            "coverage": round(len(points) / n, 3) if n else 0,
            "segments": len(segments),
        },
        "points": points,
        "segments": segments,
        "timeline": timeline,            # дорожка по аятам (смены аята во времени)
        "word_timeline": word_timeline,  # дорожка по словам (время -> слово в аяте)
    }


def _longest_forward_chain(raw: list[dict]) -> list[dict]:
    """Взвешенная наибольшая неубывающая по corpus-позиции подпоследовательность сегментов
    (в порядке времени). Вес = число точек. Допускается откат назад не более BACK_TOL."""
    k = len(raw)
    if k == 0:
        return []
    w = [s["n_points"] for s in raw]
    best = w[:]                 # лучший суммарный вес цепочки, кончающейся на i
    prev = [-1] * k
    for i in range(k):
        for j in range(i):
            # i может следовать за j, если не откатывается назад больше допуска
            if raw[i]["lo"] >= raw[j]["lo"] - BACK_TOL and best[j] + w[i] > best[i]:
                best[i] = best[j] + w[i]
                prev[i] = j
    end = max(range(k), key=lambda i: best[i])
    chain = []
    while end != -1:
        chain.append(raw[end])
        end = prev[end]
    return chain[::-1]


def _segment(points: list[dict], quran: Quran):
    segs: list[list[dict]] = []
    cur: list[dict] = []
    for p in points:
        if not cur:
            cur = [p]
            continue
        prev = cur[-1]
        di = p["asr_i"] - prev["asr_i"]
        diag_prev = prev["corpus"] - prev["asr_i"]
        diag_cur = p["corpus"] - p["asr_i"]
        same = di <= GAP_TOL and abs(diag_cur - diag_prev) <= DIAG_TOL
        if same:
            cur.append(p)
        else:
            segs.append(cur)
            cur = [p]
    if cur:
        segs.append(cur)

    # сводка по каждому сырому сегменту (+ храним точки для timeline)
    raw = []
    for s in segs:
        if len(s) < MIN_SEG_WORDS:
            continue
        c0, c1 = s[0]["corpus"], s[-1]["corpus"]
        lo, hi = min(c0, c1), max(c0, c1)
        raw.append({
            "points": s,
            "surah": quran.tokens[lo].surah,
            "n_points": len(s),
            "confidence": sum(p["support"] for p in s) / len(s),
            "lo": lo, "hi": hi,
        })

    # Монотонность позиции: чтение движется вперёд по корпусу (реальные смены суры —
    # тоже рост позиции). Оставляем самую «тяжёлую» неубывающую по corpus цепочку
    # сегментов (допуская малый откат BACK_TOL на повторы аятов). Ложные блипы —
    # это скачок назад с возвратом, они выпадают из цепочки.
    keep = _longest_forward_chain(raw)

    # timeline (по аятам) и word_timeline (по словам) — только по выжившим сегментам.
    # word_timeline делаем ПЛОТНЫМ: выровнены лишь часть ASR-слов, между якорями —
    # дыры. Идём по корпусу слово-за-словом и раздаём времена линейно между соседними
    # якорями, чтобы подсветка ехала плавно, а не залипала на последнем выровненном
    # слове и не прыгала. Остаточную неточность границ смазывает окно 2-3 слов в плеере.
    # У ЯКОРЕЙ несём реальный конец слова (t_end от распознавателя) → плеер замораживает
    # заливку на паузе, а не «протягивает» её до следующего слова через тишину; и метрика
    # coverage считает настоящую длительность речи, а не CAP-догадку. Интерполированным
    # словам t_end не даём — реального конца у них нет, плеер сам растянет их до соседа.
    out = []
    timeline = []
    word_timeline = []

    def push_word(t, tok_corpus, t_end=None):
        tok = quran.tokens[tok_corpus]
        if word_timeline and word_timeline[-1]["corpus"] == tok_corpus:
            return
        if word_timeline and t <= word_timeline[-1]["t"]:
            t = round(word_timeline[-1]["t"] + 0.001, 3)  # держим строгий рост времени
        entry = {"t": t, "surah": tok.surah, "ayah": tok.ayah,
                 "wi": tok.word_index, "corpus": tok_corpus}
        if t_end is not None and t_end > t:            # реальный конец слова (только якоря)
            entry["t_end"] = round(t_end, 3)
        word_timeline.append(entry)

    anchors = [p for seg in keep for p in seg["points"]]
    for idx, p in enumerate(anchors):
        push_word(p["t"], p["corpus"], p.get("t_end"))
        if idx + 1 < len(anchors):
            q = anchors[idx + 1]
            c0, c1, t0, t1 = p["corpus"], q["corpus"], p["t"], q["t"]
            if 2 <= (c1 - c0) <= INTERP_MAX_GAP and t1 > t0:
                span = c1 - c0
                for c in range(c0 + 1, c1):        # добиваем пропущенные слова корпуса (без t_end)
                    push_word(round(t0 + (c - c0) / span * (t1 - t0), 3), c)

    for seg in keep:
        for p in seg["points"]:
            if timeline and (timeline[-1]["surah"], timeline[-1]["ayah"]) == (p["surah"], p["ayah"]):
                timeline[-1]["t_end"] = p["t_end"]
                continue
            timeline.append({"t": p["t"], "t_end": p["t_end"],
                             "surah": p["surah"], "ayah": p["ayah"],
                             "corpus": p["corpus"]})
        t0, t1 = quran.tokens[seg["lo"]], quran.tokens[seg["hi"]]
        pts = seg["points"]
        out.append({
            "t_start": pts[0]["t"],
            "t_end": pts[-1]["t_end"],
            "surah_start": t0.surah, "ayah_start": t0.ayah,
            "surah_end": t1.surah, "ayah_end": t1.ayah,
            "surah_title": quran.surah(t0.surah).title,
            "corpus_start": seg["lo"], "corpus_end": seg["hi"],
            "n_points": seg["n_points"],
            "confidence": round(seg["confidence"], 2),
        })
    return out, timeline, word_timeline


# --- счётчики ASR↔эталон (идея quran-align match.cc) --------------------------


def match_stats(asr_norms: list[str], sync_map: dict, quran: Quran) -> dict:
    """Сопоставить ASR-слова с эталонным текстом найденного диапазона и посчитать
    hits (точные совпадения) / subs (сматчено с заменой) / ins (лишние ASR-слова:
    шум, повторы чтеца) / dels (слова эталона без ASR-опоры). wer = (subs+ins+dels)/ref.

    Эталон — корпусные слова диапазонов выживших сегментов sync-map, в порядке чтения
    (обе стороны в нормализованной форме M1). Объективная метрика «каши» распознавания,
    сравнимая между прогонами, — в отличие от самореферентного aligned_ratio.
    """
    import difflib
    ref = []
    for seg in sync_map.get("segments", []):
        ref.extend(quran.tokens[c].text
                   for c in range(seg["corpus_start"], seg["corpus_end"] + 1))
    if not ref or not asr_norms:
        return {}
    sm = difflib.SequenceMatcher(a=asr_norms, b=ref, autojunk=False)
    hits = subs = ins = dels = 0
    for op, i0, i1, j0, j1 in sm.get_opcodes():
        if op == "equal":
            hits += i1 - i0
        elif op == "replace":
            common = min(i1 - i0, j1 - j0)
            subs += common
            ins += (i1 - i0) - common      # лишний хвост ASR внутри замены
            dels += (j1 - j0) - common     # недобранный хвост эталона
        elif op == "delete":               # кусок только в ASR
            ins += i1 - i0
        elif op == "insert":               # кусок только в эталоне
            dels += j1 - j0
    return {"ref_words": len(ref), "hits": hits, "subs": subs, "ins": ins, "dels": dels,
            "wer": round((subs + ins + dels) / len(ref), 3)}


# ======================= БУКВЕННЫЙ ВХОД (w2v/mms) — ЛОКАЛИЗАЦИЯ =======================


_K = 5                    # длина k-граммы для буквенной локализации
_NEG = -1e9
_REGION_MARGIN = 6        # ± аятов запаса вокруг плотного кластера (CTC добьёт точную границу)
_REFINE_BAND = 4          # ± аятов точного difflib-добора вокруг приближённой (по префиксам) границы


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
    """Взвешенная плотность k-грамм-попаданий декода на каждый плоский аят (буквенно, дёшево).

    IDF-взвешивание: каждая k-грамма декода вносит СУММАРНО 1.0, размазанное по своим совпадениям
    (вес 1/df на попадание, df = число позиций k-граммы в Коране). Редкая (дискриминативная)
    k-грамма → вес концентрируется на немногих аятах (сильный сигнал); частая (общие фразы —
    истиаза/басмала/زачины) → размазана в пыль. Без IDF пик плотности создавали именно общие
    фразы (rec10 Ар-Рахман улетал в 33-34: там острый мусорный пик, а сура 55 размазана)."""
    dens = np.zeros(n_fa, dtype=np.float64)
    for sp in range(len(skel) - _K + 1):
        cps = kidx.get(skel[sp:sp + _K])
        if not cps:
            continue
        w = 1.0 / len(cps)
        for cp in cps:
            dens[char2fa[cp]] += w
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
               index=None, verbose: bool = False, dec: str | None = None) -> list[tuple[int, int]] | None:
    """Главный вход: список (surah, ayah) читаемого диапазона из эмиссий (по порядку). None если нет.

    Диапазон — произвольный непрерывный отрезок плоских аятов (часть суры / через границу сур —
    указка владельца). Чисто буквенно (без CTC/GPU): (1) k-грамм-плотность → плотный кластер аятов;
    (2) difflib-добор границ (совпадения+промежутки) → максимум ratio = истинное окно."""
    special = {ch2idx.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>", "|", "-", "ـ")} - {None}
    Cs, char2fa, kidx, flat_ayahs, fa_skel = index or build_index(quran)
    n_fa = len(flat_ayahs)

    if dec is None:
        dec = greedy_skeleton(emissions, idx2ch, special)
    if len(dec) < _K:
        return None

    # 1) буквенная локализация. IDF-плотность даёт ОСТРЫЙ пик на самом дискриминативном аяте
    # чтения (напр. rec10 Ар-Рахман → пик на 55:33). Регион вокруг пика берём ШИРОКО — не уже, чем
    # ВСЯ сура пика (короткоаятные суры типа Аль-Вакиа: 96 аятов, но по ~25 симв → оценка по средней
    # длине аята сильно занижала и регион обрезал старт/конец). Ширина = max(оценка по длине декода,
    # длина суры пика) + запас, с ОБЕИХ сторон пика (истинные границы гарантированно влезают; лишнее
    # обрежет добор). difflib на широком регионе — всё равно секунды.
    dens = _ayah_density(dec, char2fa, kidx, n_fa)
    if dens.sum() == 0:
        return None
    peak = int(dens.argmax())
    avg_ay = max(1.0, len(Cs) / n_fa)          # средняя длина скелета аята (симв.)
    n_est = int(len(dec) / avg_ay)             # оценка числа читаемых аятов
    s_peak = flat_ayahs[peak][0]
    s_len = len(quran.surah(s_peak).verses)    # длина суры пика (аятов)
    half = max(n_est, s_len) + _REGION_MARGIN
    lo = max(0, peak - half)
    hi = min(n_fa - 1, peak + half)
    if verbose:
        ps, pa = flat_ayahs[peak]; s0, a0 = flat_ayahs[lo]; s1, a1 = flat_ayahs[hi]
        print(f"пик плотности: {ps}:{pa}; регион {s0}:{a0}..{s1}:{a1} ({hi-lo+1} аятов, n_est={n_est})")

    # 2) границы окна. Полный O(регион²) перебор difflib НЕ масштабируется на длинные суры (целая
    # Марьям: регион ~121 аят, декод 18-мин записи → difflib на огромных строках × тысячи окон =
    # минуты-десятки минут CPU). Дёшево и точно:
    #   (а) ОДНО глобальное difflib-выравнивание декода к тексту всего региона → matching-блоки →
    #       matched-символов на каждый аят (cov[k]);
    #   (б) АППРОКСИМАЦИЯ той же метрики ratio≈2·ΣCov/(len(dec)+ΣLen) через ПРЕФИКС-СУММЫ: максимум
    #       по ВСЕМ окнам за O(регион²) чистой арифметики (микросекунды, без difflib);
    #   (в) КРОШЕЧНЫЙ добор ±_REFINE_BAND настоящей метрикой _difflib_score (десятки вызовов) —
    #       правит приближение до точного max. Результат тот же (rec5 25:63-77, rec7 6:95-103),
    #       но find_range секунды вместо минут даже на целой суре.
    region_skels = fa_skel[lo:hi + 1]
    nreg = len(region_skels)
    char2ay = []                              # char-позиция в region_text → локальный индекс аята
    for k, sk in enumerate(region_skels):
        char2ay.extend([k] * len(sk))
    region_text = "".join(region_skels)

    import difflib
    sm = difflib.SequenceMatcher(None, dec, region_text, autojunk=False)
    cov = [0] * nreg
    for b in sm.get_matching_blocks():
        for pos in range(b.b, b.b + b.size):
            cov[char2ay[pos]] += 1
    lens = [len(sk) for sk in region_skels]
    pc = [0] * (nreg + 1)                      # префикс-суммы matched (cov)
    pl = [0] * (nreg + 1)                      # префикс-суммы длин аятов
    for k in range(nreg):
        pc[k + 1] = pc[k] + cov[k]
        pl[k + 1] = pl[k] + lens[k]
    Ld = len(dec)
    # (б) приближённый максимум ratio по всем окнам (арифметика на префиксах)
    e0 = e1 = 0
    best_ar = -1.0
    for a in range(nreg):
        for b in range(a, nreg):
            m = pc[b + 1] - pc[a]
            if m == 0:
                continue
            ar = 2.0 * m / (Ld + (pl[b + 1] - pl[a]))
            if ar > best_ar:
                best_ar = ar; e0, e1 = a, b

    # (в) узкий добор точной метрикой по НЕЗАВИСИМЫМ осям (O(band), не O(band²)): при конце=e1
    # ищем лучший старт в ±B, затем при этом старте — лучший конец в ±B. На длинной суре каждый
    # _difflib_score дорог (окно ~тысячи симв.), поэтому десятки вызовов, не сотни.
    B = _REFINE_BAND
    def _score(a, b):
        return _difflib_score(dec, "".join(region_skels[a:b + 1]))
    cand0 = range(max(0, e0 - B), min(nreg, e0 + B + 1))
    c0 = max(cand0, key=lambda a: _score(a, e1))
    cand1 = [b for b in range(max(0, e1 - B), min(nreg, e1 + B + 1)) if b >= c0]
    c1 = max(cand1, key=lambda b: _score(c0, b)) if cand1 else e1
    cur = _score(c0, c1)

    # (г) добор границ ВНУТРИ ТОЙ ЖЕ суры: мелодичный/необычный зачин (напр. Марьям 19:1 كٓهيعٓصٓ —
    # разрозненные буквы, декодятся бедно → ratio их отрезает) реально читается, но у него низкий
    # cov. Тянем старт назад / конец вперёд, пока аят той же суры и его cov не ноль. НЕ пересекаем
    # границу суры (иначе ложно залезаем в предыдущую — rec9: 16:127 ловит истиазу, но чтение с 17:1).
    # тянем только В ПРЕДЕЛАХ суры ПИКА (s_peak) — надёжный якорь чтения; так не залезаем в
    # соседнюю суру по ложному cov (rec14: старт refine уже в суре 68 ≠ пик 69 → добор не бежит).
    while (c0 - 1 >= 0 and flat_ayahs[lo + c0 - 1][0] == s_peak
           and flat_ayahs[lo + c0][0] == s_peak and cov[c0 - 1] > 0):
        c0 -= 1
    while (c1 + 1 < nreg and flat_ayahs[lo + c1 + 1][0] == s_peak
           and flat_ayahs[lo + c1][0] == s_peak and cov[c1 + 1] > 0):
        c1 += 1
    i0, i1 = lo + c0, lo + c1

    # (д) срезать ОДИНОЧНЫЕ хвосты чужой суры на краях: диапазон не должен начинаться последним
    # аятом суры перед сменой суры (или кончаться первым аятом новой) — это почти всегда ложный
    # cov на границе, а не реальное кросс-сурное чтение (rec14: старт 68:52 перед 69:1 → срезать).
    while i0 < i1 and flat_ayahs[i0][0] != flat_ayahs[i0 + 1][0]:
        i0 += 1
    while i1 > i0 and flat_ayahs[i1][0] != flat_ayahs[i1 - 1][0]:
        i1 -= 1

    # (е) РОБАСТНОСТЬ К ПОВТОРЯЮЩИМСЯ СУРАМ. Приближение по префиксам (б) полагается на ОДНО
    # глобальное difflib-выравнивание декода к региону. Рефрен (Ар-Рахман 55: «فبأي آلاء ربكما
    # تكذبان» ×31) путает монотонный difflib → cov по суре пика ≈ 0 → префикс-макс уводит в соседа
    # (rec10: даёт 55:77..56:39 вместо всей суры 55:1-78). Пробуем ВТОРОЙ сид — ЦЕЛУЮ суру пика — и
    # оставляем окно с бОльшим РЕАЛЬНЫМ difflib-ratio (истинная цель дизайна; проверено: истина
    # 55:1-78 имеет ratio 0.32 — максимум среди окон). Частичное чтение сид-2 НЕ перебьёт: у целой
    # суры ratio ниже частичного окна (непрочитанные аяты раздувают знаменатель) — 8/9 не задеты.
    cur = _difflib_score(dec, "".join(fa_skel[i0:i1 + 1]))
    ps_lo = next((k for k, fa in enumerate(flat_ayahs) if fa[0] == s_peak), None)
    ps_hi = next((k for k in range(len(flat_ayahs) - 1, -1, -1) if flat_ayahs[k][0] == s_peak), None)
    if ps_lo is not None:
        rj = _difflib_score(dec, "".join(fa_skel[ps_lo:ps_hi + 1]))   # RAW вся сура (без ±B-добора)
        if rj > cur:
            i0, i1, cur = ps_lo, ps_hi, rj
    verses = flat_ayahs[i0:i1 + 1]
    if verbose:
        s0, a0 = verses[0]; s1, a1 = verses[-1]
        print(f"диапазон: {s0}:{a0}..{s1}:{a1}  difflib-ratio={cur:.3f}")
    return verses
