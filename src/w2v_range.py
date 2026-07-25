"""w2v_range — тонкий ре-экспорт БУКВЕННОЙ локализации из общего модуля `match_align` (директива
владельца 25.07: матчинг с Кораном = ОДИН переиспользуемый модуль). Реализация в
`src/match_align.py`; этот файл сохраняет исторический импорт-контракт (`build_index`/`find_range`/
`greedy_skeleton`/`ayah_start_hints`/`ctc_logprob`/`pool_emissions` + приватные хелперы) для
`gpu_align` и офлайн-проб. Новый код импортируй прямо из `match_align`.
"""
from __future__ import annotations

from match_align import (  # noqa: F401  (ре-экспорт — имена используются импортёрами)
    build_index,
    find_range,
    greedy_skeleton,
    ayah_start_hints,
    pool_emissions,
    ctc_logprob,
    _difflib_score,
    _text_to_ids,
    _ayah_density,
    _dense_region,
    _K,
    _NEG,
    _REGION_MARGIN,
    _REFINE_BAND,
)
