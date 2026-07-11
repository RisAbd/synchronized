"""Реестр распознавателей (ASR-бэкендов).

Добавить новый распознаватель = запись здесь + ветка в `pipeline._recognize`.
Порядок в REGISTRY = приоритет авто-выбора активного прогона в плеере
(первый готовый по этому порядку становится активным по умолчанию).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recognizer:
    key: str
    label: str          # человекочитаемое имя (для UI)
    note: str = ""      # короткое пояснение (точность/особенности)


# Порядок важен: ручная правка (человек) — истина, выше всех → активна по умолчанию, если есть;
# затем forced align (выравнивает известный текст) → точнее ASR; затем google (точнее whisper на
# арабском).
REGISTRY: dict[str, Recognizer] = {
    "manual": Recognizer("manual", "Ручной", "ручная привязка слов мышью (элайнер П12); человек правит поверх forced/ASR — истина, приоритет авто-выбора"),
    "forced": Recognizer("forced", "Forced align", "точные границы: выравнивает текст аятов к аудио (без ошибок ASR); нужен готовый прогон для диапазона"),
    "google": Recognizer("google", "Google STT", "точнее на арабском; из кэша ответов"),
    "whisper": Recognizer("whisper", "Whisper (Tarteel)", "локально на GPU; модель tarteel-ai/whisper-base-ar-quran — дообучена под коранический арабский"),
}

# Выравниватели по известному тексту (не распознаватели-ASR): их не выбирают при добавлении записи.
# forced берёт диапазон из готового прогона-источника (google/whisper); manual правит человек в
# ручном элайнере (П12) поверх готовой записи. Оба исключены из selectable_recognizers и из
# выбора источника forced (_forced_source).
ALIGNERS = {"forced", "manual"}

# Приоритет авто-выбора активного прогона (по убыванию предпочтения).
PRIORITY = list(REGISTRY.keys())


def is_aligner(key: str) -> bool:
    return key in ALIGNERS


def is_valid(key: str) -> bool:
    return key in REGISTRY


def label_of(key: str) -> str:
    r = REGISTRY.get(key)
    return r.label if r else key


def all_recognizers() -> list[Recognizer]:
    return list(REGISTRY.values())


def selectable_recognizers() -> list[Recognizer]:
    """Распознаватели, которые пользователь выбирает при добавлении записи.
    Без выравнивателей: forced align — не отдельный распознаватель, а автоматический
    пост-шаг после ASR (см. tasks._maybe_forced)."""
    return [r for r in REGISTRY.values() if r.key not in ALIGNERS]
