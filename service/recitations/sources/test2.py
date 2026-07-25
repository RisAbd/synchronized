"""Экспериментальный слот-прогон №2 — второй слот для опытов рядом с `test` (см. test.py)."""
KEY = "test2"
LABEL = "ТЕСТ 2"
NOTE = "экспериментальный слот №2 — результаты ручных опытов выравнивания (заполняется скриптом)"
SELECTABLE = False
AUTO = False
ISOLATE = False
ALIGNED = True
PRIORITY = 91


def available() -> bool:
    return True


def ready(rec) -> bool:
    return False


def run(rec, audio, quran, out_dir, stage=None):
    raise NotImplementedError("тестовый слот заполняется вручную (скриптом), не через run()")
