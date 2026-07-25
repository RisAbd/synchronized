"""Пакет ИСТОЧНИКОВ распознавания/выравнивания — плоские независимые плагины (директива
владельца 24-25.07: «выкинуть деление аллайнер vs распознаватель; каждый источник полностью
независим, один файл = один источник, динамический импорт»).

Референс — Wildbox `airflow-backend-analytics/dags/analytics_notifications` (файл-на-тип,
`importlib.import_module`, единый контракт). Здесь тот же паттерн: один `.py` на источник,
лоадер сканирует пакет и строит реестр из модулей. Никакого наследования между источниками,
никакого общего базового класса, никакого множества ALIGNERS — каждый источник САМ объявляет
свои свойства и умеет `run()`. Матчинг с Кораном у всех один — общий модуль `match_align`.

Контракт модуля-источника (см. любой файл рядом):
  KEY: str        — ключ (хранится в AsrRun.recognizer)
  LABEL: str      — имя для UI
  NOTE: str       — короткое пояснение
  SELECTABLE: bool — пользователь выбирает при добавлении записи (google/whisper). default False
  AUTO: bool       — авто-пост-шаг на КАЖДОЙ записи (forced/w2v). default False
  ISOLATE: bool    — гнать в отдельном GPU-процессе (gpu_align) — липкая CUDA-арена на 6ГБ. default False
  ALIGNED: bool    — выравнивает буквы/слова к ИЗВЕСТНОМУ тексту аятов → точные границы; влияет
                     на авто-выбор активного прогона и на группировку в плеере. default False
  PRIORITY: int    — порядок (меньше = выше/раньше) для авто-выбора и UI. обязателен де-факто

  def available() -> bool     — зависимости на месте (безопасно, без исключений). default True
  def ready(rec) -> bool      — доп. предусловие для авто-запуска (forced ждёт готовый ASR). default True
  def run(rec, audio, quran, out_dir, stage=None) -> dict
        — сделать работу, ЗАПИСАТЬ out_dir/sync-map.json и вернуть sync_map. ISOLATE-источники
          зовутся из подпроцесса (gpu_align), остальные — прямо из pipeline.run_one.

Свойства-константы (SELECTABLE/AUTO/ISOLATE/ALIGNED/PRIORITY) — это МЕТАДАННЫЕ источника, а не
деление на «аллайнеры» и «распознаватели»: control-flow больше нигде не ветвится по типу источника
(`is_aligner` выпилен), pipeline зовёт единый `run()`.
"""
from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

_REGISTRY: dict[str, ModuleType] | None = None


def _load() -> dict[str, ModuleType]:
    """Сканировать пакет, импортировать каждый модуль-источник (с атрибутом KEY), построить
    реестр, отсортированный по PRIORITY. Модули лёгкие (константы + def), тяжёлые импорты —
    внутри available()/run(), поэтому импорт пакета дешёвый и безопасный."""
    global _REGISTRY
    if _REGISTRY is None:
        reg: dict[str, ModuleType] = {}
        for m in pkgutil.iter_modules(__path__):
            if m.name.startswith("_"):
                continue
            mod = importlib.import_module(f"{__name__}.{m.name}")
            key = getattr(mod, "KEY", None)
            if key:
                reg[key] = mod
        _REGISTRY = dict(sorted(reg.items(),
                                key=lambda kv: getattr(kv[1], "PRIORITY", 999)))
    return _REGISTRY


def all_sources() -> list[ModuleType]:
    """Все источники в порядке PRIORITY (меньше = раньше)."""
    return list(_load().values())


def keys() -> list[str]:
    """Ключи источников в порядке PRIORITY (заменяет прежний recognizers.PRIORITY)."""
    return list(_load().keys())


def get(key: str) -> ModuleType | None:
    return _load().get(key)


def is_valid(key: str) -> bool:
    return key in _load()


def label_of(key: str) -> str:
    m = _load().get(key)
    return getattr(m, "LABEL", key) if m else key


def is_aligned(key: str) -> bool:
    """Источник выравнивает к известному тексту аятов (точные границы). Заменяет прежний
    `is_aligner`: теперь это объявленное свойство источника (ALIGNED), а не глобальный тип-тест."""
    m = _load().get(key)
    return bool(getattr(m, "ALIGNED", False)) if m else False


def is_isolated(key: str) -> bool:
    m = _load().get(key)
    return bool(getattr(m, "ISOLATE", False)) if m else False


def selectable() -> list[ModuleType]:
    """Источники, выбираемые пользователем при добавлении записи (google/whisper)."""
    return [m for m in all_sources() if getattr(m, "SELECTABLE", False)]


def auto() -> list[ModuleType]:
    """Авто-источники (forced/w2v) — запускаются на каждой записи пост-шагом после ASR."""
    return [m for m in all_sources() if getattr(m, "AUTO", False)]


def available(mod: ModuleType) -> bool:
    """Безопасно проверить наличие зависимостей источника (any exception → недоступен)."""
    try:
        fn = getattr(mod, "available", None)
        return bool(fn()) if fn else True
    except Exception:
        return False


def ready(mod: ModuleType, rec) -> bool:
    """Доп. предусловие авто-запуска (forced: есть готовый ASR-источник диапазона)."""
    try:
        fn = getattr(mod, "ready", None)
        return bool(fn(rec)) if fn else True
    except Exception:
        return False
