"""M1 — канонический текст Корана: загрузка, нормализация, плоский корпус токенов.

Чистый модуль без внешнего I/O кроме чтения quran.db. От него зависят align и player.

Идея: помимо оригинального текста (с харакатами) строим единый «поисковый корпус» —
непрерывный поток нормализованных слов с обратным маппингом
    global_index -> (surah, ayah, word_index_in_ayah).
Это фундамент для выравнивания ASR-транскрипции на канон (M4 align).

Нормализация арабского приводит текст к «согласному скелету», устойчивому к тому,
как чтение распознаёт ASR: снимаются харакаты и кораническая разметка, схлопываются
варианты алифа/хамзы/я/та-марбуты.

CLI:  python3 quran.py           # статистика + примеры
"""
from __future__ import annotations

import sqlite3
import unicodedata
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "quran.db"

# --- нормализация -----------------------------------------------------------

# всё, что выбрасываем: харакаты (U+064B–U+0652), дагер-алиф (U+0670),
# кораническая разметка/паузы (U+06D6–U+06DC), знак саджды (U+06E9),
# tatweel (U+0640, в БД не встречается, но на всякий случай).
_STRIP = (
    set(range(0x064B, 0x0653))  # tashkeel
    | {0x0670}                  # superscript (dagger) alef
    | set(range(0x06D6, 0x06DD))  # small high signs / pause marks
    | {0x06E9}                  # sajdah sign
    | {0x0640}                  # tatweel
)

# свёртка вариантов букв в единую форму (агрессивно — под fuzzy-матчинг с ASR):
_FOLD = {
    "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا", "ٱ": "ا",
    "ى": "ي",
    "ؤ": "و",
    "ئ": "ي",
    "ة": "ه",
    "ء": "",   # одиночную хамзу убираем
}

_STRIP_TABLE = {cp: None for cp in _STRIP}
_FOLD_TABLE = {ord(k): (v or None) for k, v in _FOLD.items()}


def normalize(text: str) -> str:
    """Оригинальный арабский → нормализованный согласный скелет (слова через один пробел)."""
    text = text.translate(_STRIP_TABLE)
    text = text.translate(_FOLD_TABLE)
    return " ".join(text.split())


def skeleton(word: str) -> str:
    """Чистый согласный скелет слова: снять ВСЕ комбинирующие знаки (харакат, вакфы, дагер-алиф,
    мадда, U+06EA и т.п. — НЕЗАВИСИМО от редакции) + свернуть варианты букв. В отличие от
    `normalize` (её фикс-список `_STRIP` не покрывает диянетовские знаки — U+06EA, мадда U+0653,
    U+06EC…), опирается на `unicodedata.combining`, поэтому даёт ОДИНАКОВЫЙ скелет для одного и
    того же слова в разных редакциях (Tanzil/Diyanet). Нужен для маппинга редакций (`map_editions`)."""
    base = "".join(c for c in word if not unicodedata.combining(c))
    return base.translate(_FOLD_TABLE)


def map_editions(a_words: list[str], b_words: list[str]) -> list[list[int]]:
    """Сопоставить слова двух редакций одного аята по согласному скелету.

    Обе редакции — один текст, отличаются диакритикой и МЕСТАМИ дроблением слов (Diyanet сливает
    ذو+مره→ذومره; у Tanzil бывают отдельные токены-вакфы, дающие пустой скелет). Возвращает для
    КАЖДОГО слова `a_words[i]` список индексов слов `b_words`, которым оно соответствует
    (обычно [j]; при слиянии/дроблении — [] или несколько). Двухуказательное выравнивание:
    равные скелеты → 1:1; иначе пробуем слияние (a[i] = b[j]+b[j+1]+…) или дробление
    (b[j] = a[i]+a[i+1]+…); пустой скелет (вакф) не мапим; при рассинхроне — 1:1 fail-safe."""
    A = [skeleton(w) for w in a_words]
    B = [skeleton(w) for w in b_words]
    res: list[list[int]] = [[] for _ in a_words]
    na, nb = len(A), len(B)
    i = j = 0
    while i < na and j < nb:
        if A[i] == "":
            i += 1; continue          # пустой токен (вакф) в A — ни к чему не мапим
        if B[j] == "":
            j += 1; continue
        if A[i] == B[j]:
            res[i].append(j); i += 1; j += 1; continue
        # слияние: одно слово A = несколько слов B
        if A[i].startswith(B[j]):
            acc, k = B[j], j + 1
            while k < nb and acc != A[i] and A[i].startswith(acc + B[k]):
                acc += B[k]; k += 1
            if acc == A[i]:
                res[i] = list(range(j, k)); i += 1; j = k; continue
        # дробление: несколько слов A = одно слово B
        if B[j].startswith(A[i]):
            acc, k = A[i], i + 1
            while k < na and acc != B[j] and B[j].startswith(acc + A[k]):
                acc += A[k]; k += 1
            if acc == B[j]:
                for x in range(i, k):
                    res[x] = [j]
                i = k; j += 1; continue
        # рассинхрон (нет 1:1/слияния/дробления). Решаем по lookahead, что лишнее: напр. у Tanzil
        # к 1-му аяту суры приклеена басмала (4 слова), которых нет у Diyanet → B[j] найдётся дальше
        # в A → пропускаем A[i] (не мапим, res[i]=[]). Симметрично для лишнего слова в B.
        LA = 6
        if B[j] in A[i + 1:i + 1 + LA]:
            i += 1                          # A[i] лишнее (басмала и т.п.) — оставляем res[i]=[]
        elif A[i] in B[j + 1:j + 1 + LA]:
            j += 1                          # B[j] лишнее
        else:
            res[i].append(j); i += 1; j += 1   # настоящий рассинхрон — 1:1 и дальше (без зацикливания)
    return res


# --- модели -----------------------------------------------------------------


@dataclass
class Verse:
    surah: int          # номер суры (1..114)
    ayah: int           # номер аята внутри суры
    text: str           # оригинал с харакатами (редакция Tanzil)
    sacdah: bool = False
    text_diyanet: str = ""   # редакция турецкого мусхафа Diyanet (П9); "" если не импортирована

    @cached_property
    def norm(self) -> str:
        return normalize(self.text)

    @cached_property
    def words(self) -> list[str]:
        return self.norm.split()

    @property
    def ref(self) -> str:
        return f"{self.surah}:{self.ayah}"


@dataclass
class Surah:
    number: int
    title: str
    verses_count: int
    revelation_place: str
    bismillah_pre: bool
    verses: list[Verse] = field(default_factory=list)


@dataclass(frozen=True)
class Token:
    """Одно нормализованное слово корпуса + его адрес в каноне."""
    text: str            # нормализованное слово
    surah: int
    ayah: int
    word_index: int      # индекс слова внутри аята (с 0)
    global_index: int    # индекс в плоском корпусе (с 0)

    @property
    def ref(self) -> str:
        return f"{self.surah}:{self.ayah}:{self.word_index}"


# --- корпус -----------------------------------------------------------------


class Quran:
    def __init__(self, surahs: list[Surah]):
        self.surahs = surahs
        self._by_number = {s.number: s for s in surahs}

        verses: list[Verse] = []
        tokens: list[Token] = []
        for s in surahs:
            for v in s.verses:
                verses.append(v)
                for wi, w in enumerate(v.words):
                    tokens.append(
                        Token(text=w, surah=v.surah, ayah=v.ayah,
                              word_index=wi, global_index=len(tokens))
                    )
        self.verses = verses
        self.tokens = tokens

    # ---- загрузка ----
    @classmethod
    def load(cls, db_path: str | Path = DEFAULT_DB) -> "Quran":
        db_path = Path(db_path)
        if not db_path.is_file():
            raise FileNotFoundError(
                f"quran.db не найден: {db_path}. Забрать: "
                "curl 'https://raw.githubusercontent.com/RisAbd/quran-data/"
                "refs/heads/master/qurandatabase.org/quran.db' -o quran.db"
            )
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            srows = conn.execute(
                "select id, number, title, verses_count, revelation_place, "
                "bismillah_pre from surahs order by number"
            ).fetchall()
            # редакция Diyanet (П9) — опциональная колонка: в старых БД её нет
            has_diyanet = any(c[1] == "text_diyanet"
                              for c in conn.execute("PRAGMA table_info(surah_verses)"))
            cols = "surah_id, number, text, sacdah" + (", text_diyanet" if has_diyanet else "")
            vrows = conn.execute(
                f"select {cols} from surah_verses order by surah_id, number"
            ).fetchall()

        verses_by_surah_id: dict[int, list[Verse]] = {}
        # порядок сур по id совпадает с number в этой БД, но не полагаемся —
        # сгруппируем по surah_id и привяжем номер суры через map id->number.
        id_to_number = {r["id"]: r["number"] for r in srows}
        for r in vrows:
            sid = r["surah_id"]
            verses_by_surah_id.setdefault(sid, []).append(
                Verse(surah=id_to_number[sid], ayah=r["number"],
                      text=r["text"], sacdah=bool(r["sacdah"]),
                      text_diyanet=(r["text_diyanet"] if has_diyanet else "") or "")
            )

        surahs = [
            Surah(number=r["number"], title=r["title"],
                  verses_count=r["verses_count"],
                  revelation_place=r["revelation_place"],
                  bismillah_pre=bool(r["bismillah_pre"]),
                  verses=verses_by_surah_id.get(r["id"], []))
            for r in srows
        ]
        return cls(surahs)

    # ---- доступ ----
    def surah(self, number: int) -> Surah:
        return self._by_number[number]

    def verse(self, surah: int, ayah: int) -> Verse:
        return self._by_number[surah].verses[ayah - 1]

    def locate(self, global_index: int) -> Token:
        """Обратный маппинг: позиция в корпусе -> адрес в каноне."""
        return self.tokens[global_index]

    @cached_property
    def corpus_text(self) -> str:
        """Весь Коран как один нормализованный поток слов (для поиска/выравнивания)."""
        return " ".join(t.text for t in self.tokens)

    def __repr__(self) -> str:
        return f"<Quran surahs={len(self.surahs)} verses={len(self.verses)} tokens={len(self.tokens)}>"


# --- демо -------------------------------------------------------------------

if __name__ == "__main__":
    q = Quran.load()
    print(q)
    print(f"corpus chars: {len(q.corpus_text):,}")

    print("\n--- Фатиха (1:1) ---")
    v = q.verse(1, 1)
    print("оригинал :", v.text)
    print("норма    :", v.norm)
    print("слова    :", v.words)

    print("\n--- корпус: первые 12 токенов ---")
    for t in q.tokens[:12]:
        print(f"  #{t.global_index:<3} {t.ref:<8} {t.text}")

    print("\n--- locate(roundtrip) ---")
    t = q.tokens[10000]
    print(f"  token #{t.global_index} = {t.text!r} @ {t.ref} "
          f"(сура {q.surah(t.surah).title})")
