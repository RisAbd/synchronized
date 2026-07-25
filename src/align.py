"""align — тонкий ре-экспорт СЛОВЕСНОГО матчинга из общего модуля `match_align` (директива
владельца 25.07: матчинг с Кораном = ОДИН переиспользуемый модуль). Здесь ничего не живёт —
реализация в `src/match_align.py`; этот файл лишь сохраняет исторический импорт-контракт
(`align.align`/`load_transcript`/`match_stats`/`CorpusIndex`/`Word` + параметры) для pipeline,
`src/run.py` и офлайн-проб, чтобы их не править. Новый код импортируй прямо из `match_align`.
"""
from __future__ import annotations

from match_align import (  # noqa: F401  (ре-экспорт — имена используются импортёрами)
    Word,
    CorpusIndex,
    load_transcript,
    align,
    match_stats,
    normalize,
    Quran,
    WINDOW,
    MIN_SUPPORT,
    DIAG_TOL,
    GAP_TOL,
    MIN_SEG_WORDS,
    BACK_TOL,
    INTERP_MAX_GAP,
)
