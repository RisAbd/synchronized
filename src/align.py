"""M4 — выравнивание ASR-транскрипции на канонический текст Корана.

Задача: по потоку распознанных слов (с таймингами) определить, какому месту канона
соответствует каждый момент времени — устойчиво к шуму ASR, повторам чтеца и СМЕНАМ СУРЫ.

Метод (seed-and-consensus, аналог seed-and-extend из биоинформатики):
  1. Нормализуем ASR-слова тем же нормализатором, что и канон (M1).
  2. Инвертированный индекс канона: слово -> позиции в корпусе.
  3. Якоря — биграммные совпадения: ASR-биграмма (a[i], a[i+1]) == корпус-биграмма
     (c[p], c[p+1]). Биграммы отсекают львиную долю ложных одиночных совпадений.
  4. Для каждого ASR-слова берём локальное окно якорей и голосуем за «диагональ»
     d = corpus_pos - asr_pos. Доминирующая диагональ в окне даёт предсказанную
     позицию в корпусе. Смена суры = скачок диагонали; повтор чтеца = локальный сдвиг.
  5. Точки склеиваем в сегменты (непрерывные пассажи); разрывы диагонали = переходы.

Вход: transcript — список слов [{word, start, end}] (см. adapters ниже).
Выход: sync-map — {points:[...], segments:[...]} (контракт для плеера M5).

CLI:  python3 align.py <gstt_response.json | transcript.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from quran import Quran, normalize

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
    out = []
    timeline = []
    word_timeline = []

    def push_word(t, tok_corpus):
        tok = quran.tokens[tok_corpus]
        if word_timeline and word_timeline[-1]["corpus"] == tok_corpus:
            return
        if word_timeline and t <= word_timeline[-1]["t"]:
            t = round(word_timeline[-1]["t"] + 0.001, 3)  # держим строгий рост времени
        word_timeline.append({"t": t, "surah": tok.surah, "ayah": tok.ayah,
                              "wi": tok.word_index, "corpus": tok_corpus})

    anchors = [p for seg in keep for p in seg["points"]]
    for idx, p in enumerate(anchors):
        push_word(p["t"], p["corpus"])
        if idx + 1 < len(anchors):
            q = anchors[idx + 1]
            c0, c1, t0, t1 = p["corpus"], q["corpus"], p["t"], q["t"]
            if 2 <= (c1 - c0) <= INTERP_MAX_GAP and t1 > t0:
                span = c1 - c0
                for c in range(c0 + 1, c1):        # добиваем пропущенные слова корпуса
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


# --- демо/валидация ---------------------------------------------------------


def _fmt_t(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/abd/development/speech-to-text-python/"
        "gcloud-speech-data/aswailis_isra/gstt_response.json"
    )
    q = Quran.load()
    tr = load_transcript(src)
    res = align(tr, q)

    print(f"источник: {src}")
    print(f"ASR-слов: {res['meta']['asr_words']}, "
          f"привязано: {res['meta']['aligned_points']} "
          f"({res['meta']['coverage']*100:.0f}%), "
          f"сегментов: {res['meta']['segments']}\n")

    print("СЕГМЕНТЫ (пассажи):")
    for s in res["segments"]:
        rng = (f"{s['surah_start']}:{s['ayah_start']}"
               if (s['surah_start'], s['ayah_start']) == (s['surah_end'], s['ayah_end'])
               else f"{s['surah_start']}:{s['ayah_start']} → {s['surah_end']}:{s['ayah_end']}")
        print(f"  [{_fmt_t(s['t_start'])}-{_fmt_t(s['t_end'])}] "
              f"сура {s['surah_start']:>3} {s['surah_title']:<18} {rng:<20} "
              f"слов={s['n_points']:<3} conf={s['confidence']}")

    out_path = Path(__file__).resolve().parent.parent / "work" / "sync-map.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\nsync-map -> {out_path}")
