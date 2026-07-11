"""Тесты M1. Запуск: python3 test_quran.py   (или pytest test_quran.py)"""
from quran import Quran, normalize, skeleton, map_editions


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


def test_bismillah_flags_per_edition():
    """Флаги «нужна ли ОТДЕЛЬНАЯ ﷽» — per-edition (текст редакций не трогаем).
    Tanzil (`text`): басмала вшита в текст → доп. не нужна нигде (везде False).
    Diyanet (`text_diyanet`): басмалы в тексте нет → True у 112, кроме Фатихи(1) и Тавбы(9)."""
    q = Quran.load()
    # Tanzil: доп. басмала не нужна ни у одной суры
    assert all(not s.bismillah_pre for s in q.surahs)
    # Diyanet: доп. басмала нужна везде, кроме [1, 9]
    no_diy = sorted(s.number for s in q.surahs if not s.bismillah_pre_diyanet)
    assert no_diy == [1, 9]
    yes_diy = [s.number for s in q.surahs if s.bismillah_pre_diyanet]
    assert len(yes_diy) == 112
    # Тавба (9) — НИКОГДА не True (басмалы у неё нет вовсе)
    s9 = q.surah(9)
    assert not s9.bismillah_pre and not s9.bismillah_pre_diyanet


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


# --- skeleton / map_editions (маппинг редакций Tanzil↔Diyanet, П9) ---

def test_skeleton_strips_diyanet_marks():
    # диянетовские знаки (U+06EA в رَح۪يم, дагер-алиф в اللّٰه) снимаются → как у Tanzil
    assert skeleton("الرَّح۪يمِ") == skeleton("الرَّحِيمِ") == "الرحيم"
    assert skeleton("اللّٰهِ") == skeleton("اللَّهِ") == "الله"
    assert skeleton("الْهَوٰىۜ") == skeleton("الْهَوَىٰ") == "الهوي"   # ى→ي, дагер-алиф+вакф долой


def test_map_editions_identity():
    # равные по скелету слова → 1:1
    assert map_editions(["بِسْمِ", "اللَّهِ"], ["بِسْمِ", "اللّٰهِ"]) == [[0], [1]]


def test_map_editions_merge():
    # 53:6: Diyanet слил ذو+مره → ذومره; оба Tanzil-слова указывают на слитое b[0]
    a = ["ذُو", "مِرَّةٍ", "فَاسْتَوَىٰ"]           # Tanzil (3)
    b = ["ذُومِرَّةٍ", "فَاسْتَوٰى"]                 # Diyanet (2, слито)
    assert map_editions(a, b) == [[0], [0], [1]]


def test_map_editions_split():
    # обратное: одно Tanzil-слово = два Diyanet-слова
    a = ["ذُومِرَّةٍ", "فَاسْتَوَىٰ"]
    b = ["ذُو", "مِرَّةٍ", "فَاسْتَوٰى"]
    assert map_editions(a, b) == [[0, 1], [2]]


def test_map_editions_skips_empty_waqf_token():
    # у Tanzil лишний отдельный токен-вакф (пустой скелет) → не мапится, остальные 1:1
    a = ["إِنْ", "ۛ", "هِيَ"]
    b = ["اِنْ", "هِيَ"]
    assert map_editions(a, b) == [[0], [], [1]]


def test_map_editions_skips_prepended_basmala():
    # у Tanzil к аяту 1 суры приклеена басмала (4 слова), у Diyanet её нет → эти 4 слова не мапятся,
    # реальные слова аята ложатся 1:1 (кейс rec11 53:1)
    a = ["بِسْمِ", "اللَّهِ", "الرَّحْمَٰنِ", "الرَّحِيمِ", "وَالنَّجْمِ", "إِذَا", "هَوَىٰ"]
    b = ["وَالنَّجْمِ", "اِذَا", "هَوٰىۙ"]
    m = map_editions(a, b)
    assert m[:4] == [[], [], [], []]          # басмала не сматчена
    assert m[4:] == [[0], [1], [2]]           # реальные слова аята — 1:1


def test_map_editions_real_db_najm():
    # на реальных данных: каждое НЕпустое слово Tanzil получает хотя бы один индекс Diyanet
    import sqlite3, pathlib
    db = pathlib.Path(__file__).resolve().parent.parent / "data" / "quran.db"
    con = sqlite3.connect(db)
    row = con.execute("select text, text_diyanet from surah_verses where surah_id=53 and number=6").fetchone()
    con.close()
    if not row or not row[1]:
        return  # text_diyanet ещё не импортирован — тест пропускаем
    tw, dw = row[0].split(), row[1].split()
    m = map_editions(tw, dw)
    for i, w in enumerate(tw):
        if skeleton(w):
            assert m[i], f"слово {i} ({w}) не сматчено"


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
