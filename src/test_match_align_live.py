"""Тесты live-локатора (WI): locate / StreamLocator / SegmentTracker — ЧИСТЫЕ функции поиска места
в Коране, БЕЗ модели/GPU/эмиссий. Синтетический мини-индекс формата `_index_over` (Cs, char2fa,
kidx, flat_ayahs, fa_skel). Гоняет: k-грамм-голосование находит верный аят; band ограничивает поиск;
StreamLocator лочится/трекает; SegmentTracker монотонно ползёт по мини-корпусу и склеивает НЕСМЕЖНЫЕ
сегменты (мульти-сегмент Фатиха→Исра).

Запуск: python3 test_match_align_live.py   (или pytest)
"""
from collections import defaultdict

from match_align import locate, StreamLocator, SegmentTracker, _K


def _mk_index(ayahs):
    """ayahs = [(surah, ayah, skel)] → 5-кортеж индекса как у `_index_over` (k=_K)."""
    flat_ayahs, fa_skel, C, char2fa = [], [], [], []
    for fa, (s, a, sk) in enumerate(ayahs):
        flat_ayahs.append((s, a))
        fa_skel.append(sk)
        for ch in sk:
            C.append(ch); char2fa.append(fa)
    Cs = "".join(C)
    kidx = defaultdict(list)
    for p in range(len(Cs) - _K + 1):
        kidx[Cs[p:p + _K]].append(p)
    return Cs, char2fa, kidx, flat_ayahs, fa_skel


# распознаваемо-разные скелеты (длиннее _K и minblk), несмежные суры для мульти-сегмента
AYAHS = [
    (1, 1, "alhamdulillahi"),
    (1, 2, "arrahmanirrahim"),
    (1, 3, "malikiyawmiddin"),
    (17, 1, "subhanallathiasra"),
    (17, 2, "waataynamusalkitab"),
]
IDX = _mk_index(AYAHS)


# --- locate -------------------------------------------------------------------

def test_locate_finds_ayah():
    r = locate("alhamdulillahi", IDX)
    assert r is not None and (r["surah"], r["ayah"]) == (1, 1), r

def test_locate_finds_middle_ayah():
    r = locate("malikiyawmiddin", IDX)
    assert (r["surah"], r["ayah"]) == (1, 3), r

def test_locate_none_on_garbage():
    assert locate("zzzzzzzzz", IDX) is None

def test_locate_band_restricts():
    # окно текста аята (17,1) [fa=3], но prior у (1,1) [fa=0], band ahead малый → в band не попадает
    r = locate("subhanallathiasra", IDX, prior_fa=0, back=0, ahead=1)
    # либо None (в band нет сигнала), либо НЕ (17,1) — band не пустил далеко вперёд
    assert r is None or (r["surah"], r["ayah"]) != (17, 1), r

def test_locate_band_allows_forward():
    r = locate("arrahmanirrahim", IDX, prior_fa=0, back=0, ahead=3)  # (1,2) fa=1 в band
    assert (r["surah"], r["ayah"]) == (1, 2), r


# --- StreamLocator ------------------------------------------------------------

def test_stream_locks_and_tracks():
    trk = StreamLocator(IDX, conf_lock=0.0)   # conf_lock=0 → лочится на первом сигнале
    r = trk.feed("alhamdulillahi")
    assert r and (r["surah"], r["ayah"]) == (1, 1) and r["locked"]
    r = trk.feed("arrahmanirrahim")
    assert (r["surah"], r["ayah"]) == (1, 2), r

def test_stream_no_lock_on_garbage():
    trk = StreamLocator(IDX, conf_lock=0.99)
    assert trk.feed("zzzzzzzzz") is None
    assert not trk.locked


# --- SegmentTracker (онлайн-указатель по пассажу) -----------------------------

def test_segment_tracker_monotonic():
    # реалистично: хвостовое окно заканчивается ВНУТРИ читаемого аята → позиция = этот аят
    verses = [(1, 1), (1, 2), (1, 3)]
    trk = SegmentTracker(IDX, verses, minblk=6)
    p0 = trk.p
    r = trk.feed("alhamdul")                 # внутри (1,1)
    assert (r["surah"], r["ayah"]) == (1, 1), r
    p1 = trk.p
    r = trk.feed("kiyawmi")                   # внутри (1,3) "malikiyawmiddin"
    assert (r["surah"], r["ayah"]) == (1, 3), r
    assert trk.p >= p1 >= p0                  # указатель монотонно НЕ убывает

def test_segment_tracker_multiseg_contiguous():
    # НЕсмежные суры 1 и 17 склеиваются в мини-корпусе → указатель проходит без глобального прыжка
    verses = [(1, 1), (17, 1), (17, 2)]
    trk = SegmentTracker(IDX, verses, minblk=6)
    trk.feed("alhamdul")
    r = trk.feed("hanallathi")                # внутри (17,1) "subhanallathiasra"
    assert (r["surah"], r["ayah"]) == (17, 1), r
    r = trk.feed("ynamusalk")                 # внутри (17,2) "waataynamusalkitab"
    assert (r["surah"], r["ayah"]) == (17, 2), r

def test_segment_tracker_empty_verses():
    trk = SegmentTracker(IDX, [])
    assert trk.feed("alhamdulillahi") is None


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn()
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
        else:
            ok += 1
            print(f"ok   {fn.__name__}")
    print(f"\n{ok}/{len(fns)} passed")
