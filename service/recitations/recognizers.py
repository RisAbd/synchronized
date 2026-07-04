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


# Порядок важен: google точнее на арабском → выше приоритет авто-выбора.
REGISTRY: dict[str, Recognizer] = {
    "google": Recognizer("google", "Google STT", "точнее на арабском; из кэша ответов"),
    "whisper": Recognizer("whisper", "Whisper large-v3", "локально; в докере на CPU (медленно), GPU — на хосте; арабский средне"),
}

# Приоритет авто-выбора активного прогона (по убыванию предпочтения).
PRIORITY = list(REGISTRY.keys())


def is_valid(key: str) -> bool:
    return key in REGISTRY


def label_of(key: str) -> str:
    r = REGISTRY.get(key)
    return r.label if r else key


def all_recognizers() -> list[Recognizer]:
    return list(REGISTRY.values())
