"""Тесты M1. Запуск: python3 test_quran.py   (или pytest test_quran.py)"""
from quran import Quran, normalize


def test_normalize_strips_harakat():
    assert normalize("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ") == "بسم الله الرحمن الرحيم"


def test_normalize_folds_alef_variants():
    # آ أ إ ٱ -> ا
    assert normalize("أَحَد") == "احد"
    assert normalize("إِنَّ") == "ان"
    assert normalize("ٱللَّه") == "الله"


def test_normalize_folds_letters():
    assert normalize("مُوسَىٰ") == "موسي"      # ى -> ي, дагер-алиф снят
    assert normalize("ٱلصَّلَاة") == "الصلاه"   # ة -> ه
    assert normalize("شَيْءٍ") == "شي"          # одиночная хамза убрана


def test_normalize_drops_annotation_marks():
    # 2:2 содержит паузные знаки ۛ
    assert normalize("رَيْبَ ۛ فِيهِ ۛ هُدًى") == "ريب فيه هدي"


def test_normalize_collapses_whitespace():
    assert normalize("  الله    اكبر  ") == "الله اكبر"


def test_load_completeness():
    q = Quran.load()
    assert len(q.surahs) == 114
    assert len(q.verses) == 6236
    assert q.surah(2).verses_count == 286
    assert len(q.surah(2).verses) == 286


def test_bismillah_flag():
    q = Quran.load()
    # только Фатиха (1) и Тауба (9) без пре-басмалы
    no_bismillah = [s.number for s in q.surahs if not s.bismillah_pre]
    assert sorted(no_bismillah) == [1, 9]


def test_token_corpus_and_locate():
    q = Quran.load()
    assert len(q.tokens) == sum(len(v.words) for v in q.verses)
    # первый токен — начало Фатихи
    t0 = q.tokens[0]
    assert (t0.surah, t0.ayah, t0.word_index, t0.text) == (1, 1, 0, "بسم")
    # roundtrip locate
    t = q.tokens[5000]
    assert q.locate(t.global_index) is t
    assert q.locate(5000).global_index == 5000


def test_token_addresses_are_consistent():
    q = Quran.load()
    # адрес токена должен указывать на реальное слово в аяте
    for t in (q.tokens[0], q.tokens[100], q.tokens[-1]):
        v = q.verse(t.surah, t.ayah)
        assert v.words[t.word_index] == t.text


def test_corpus_text_matches_tokens():
    q = Quran.load()
    assert q.corpus_text.split() == [t.text for t in q.tokens]
    assert "  " not in q.corpus_text  # нет двойных пробелов / пустых токенов


def test_no_empty_tokens():
    q = Quran.load()
    assert all(t.text for t in q.tokens)


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
