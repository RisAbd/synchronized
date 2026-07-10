"""Тесты falign (чистые функции, БЕЗ загрузки MMS-модели).

Запуск: python3 test_falign.py   (или pytest test_falign.py)

Покрывает правило плотности и растяжку скомканных слов по дыре
(`_stuffed_runs` / `_respace_stuffed`, task2 idea 1). Сценарии заточены под баг
rec11 (Ан-Наджм 53:3→4): долгий мadd → CTC под-аллоцирует слово, следующие
скомканы; раньше их роняли (дыра+залипание), теперь растягиваем по дыре.
"""
from falign import _stuffed_runs, _respace_stuffed, MIN_WORD_SEC, STUFFED_RUN


def _ref(n):
    """Плоский ref [(surah, ayah, wi, arabic)] из n слов равной длины скелета (2 буквы)."""
    return [(53, 4, i, "اب" ) for i in range(n)]


# --- _stuffed_runs -------------------------------------------------------------

def test_stuffed_runs_detects_run():
    # 3 подряд слова < MIN_WORD_SEC = скомканный прогон
    b = [(0.0, 1.0), (1.0, 1.02), (1.02, 1.04), (1.04, 1.06), (2.0, 3.0)]
    assert _stuffed_runs(b) == [(1, 3)]


def test_stuffed_runs_ignores_isolated_short():
    # одиночные/парные короткие слова не трогаем (нужно >= STUFFED_RUN подряд)
    b = [(0.0, 1.0), (1.0, 1.02), (1.5, 2.5), (2.5, 2.52), (2.52, 2.54), (3.0, 4.0)]
    assert _stuffed_runs(b) == []  # ни одного прогона длиной >= 3


def test_stuffed_runs_multiple_and_boundary():
    # прогон в самом конце тоже ловится
    b = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.03), (1.0, 2.0),
         (2.0, 2.01), (2.01, 2.02), (2.02, 2.03), (2.03, 2.04)]
    assert _stuffed_runs(b) == [(0, 2), (4, 7)]


def test_stuffed_runs_exact_threshold():
    assert STUFFED_RUN == 3
    b = [(0.0, 0.01)] * 2 + [(1.0, 2.0)]      # 2 подряд — мало
    assert _stuffed_runs(b) == []
    b = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.03)]  # ровно 3
    assert _stuffed_runs(b) == [(0, 2)]


# --- _respace_stuffed ----------------------------------------------------------

def test_respace_rec11_like_onset_holds_prev_word():
    """rec11: الهوى (anchor, до 13.72) держится сквозь мadd; CTC поставил скомканный
    53:4 около 17.0 → раскладка стартует с 17.0 (onset), не с 13.72."""
    bounds = [
        (13.48, 13.72),                              # 0: الهوى (anchor, мadd тянется дальше)
        (17.00, 17.02), (17.02, 17.04),              # 1,2: скомканы (CTC ~17.0)
        (17.04, 17.06), (17.06, 17.08),              # 3,4: скомканы
        (20.16, 20.58),                              # 5: يوحى (anchor)
    ]
    ref = _ref(6)
    runs = _stuffed_runs(bounds)
    assert runs == [(1, 4)]
    out, cnt = _respace_stuffed(bounds, ref, runs)
    assert cnt == 4
    assert out[0] == (13.48, 13.72)                  # anchor не тронут
    assert out[5] == (20.16, 20.58)                  # anchor не тронут
    assert abs(out[1][0] - 17.00) < 1e-6             # onset = CTC-позиция куска (не 13.72)
    assert abs(out[4][1] - 20.16) < 1e-6             # конец раскладки = старт следующего anchor
    # монотонность и отсутствие нахлёстов внутри прогона
    ts = [out[i][0] for i in range(1, 5)] + [out[4][1]]
    assert ts == sorted(ts)
    for i in range(1, 4):
        assert out[i][1] == out[i + 1][0]            # стык слов встык


def test_respace_falls_back_to_full_gap_when_tight():
    """Если от CTC-позиции куска до след. слова места мало (< N×MIN_WORD_SEC),
    берём всю дыру от конца пред. слова."""
    bounds = [
        (0.0, 1.0),                                  # anchor
        (5.90, 5.91), (5.91, 5.92), (5.92, 5.93),    # 3 скомканы у самого конца дыры
        (6.0, 7.0),                                  # anchor (места от 5.90 всего 0.1 < 0.3)
    ]
    ref = _ref(5)
    out, cnt = _respace_stuffed(bounds, ref, _stuffed_runs(bounds))
    assert cnt == 3
    assert abs(out[1][0] - 1.0) < 1e-6               # onset откатился к концу пред. слова (1.0)
    assert abs(out[3][1] - 6.0) < 1e-6


def test_respace_proportional_to_skeleton_length():
    """Время дыры делится пропорционально длине согласного скелета слова."""
    bounds = [(0.0, 1.0), (10.0, 10.01), (10.01, 10.02), (10.02, 10.03), (13.0, 14.0)]
    # onset = max(1.0, 10.0) = 10.0; R = 13.0; span = 3.0
    ref = [(53, 4, 0, "ابجد"), (53, 4, 1, "اب"),  # это anchor(0) и первое скомканное
           (53, 4, 2, "ابجدهو"), (53, 4, 3, "اب"), (53, 4, 4, "اب")]
    # скелеты скомканных (индексы 1,2,3): "اб"=2, "ابجدهو"=6, "аб"=2 → доли 2:6:2 = 0.6:1.8:0.6
    out, cnt = _respace_stuffed(bounds, ref, _stuffed_runs(bounds))
    assert cnt == 3
    d1 = out[1][1] - out[1][0]
    d2 = out[2][1] - out[2][0]
    d3 = out[3][1] - out[3][0]
    assert abs(d1 - 0.6) < 1e-6
    assert abs(d2 - 1.8) < 1e-6
    assert abs(d3 - 0.6) < 1e-6
    assert abs((d1 + d2 + d3) - 3.0) < 1e-6          # покрыли всю дыру


def test_respace_noop_without_runs():
    bounds = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    out, cnt = _respace_stuffed(bounds, _ref(3), _stuffed_runs(bounds))
    assert cnt == 0
    assert out == bounds


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
