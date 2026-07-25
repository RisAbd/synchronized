"""Экспериментальный слот-прогон №1 (владелец, 25.07): сюда кладу результаты опытов выравнивания
(напр. forced-Viterbi по РУЧНОМУ расширенному тексту повторов), чтобы в плеере тыкать ОТДЕЛЬНО от
w2v и не мешать авто-прогоны. Заполняется ВРУЧНУЮ (скриптом через pipeline.build_manual_run), НЕ
авто-пост-шаг: AUTO=False, SELECTABLE=False. ALIGNED=True — группируется с выравнивателями."""
KEY = "test"
LABEL = "ручной+авто-время"
NOTE = "структура из ручного эталона владельца + автоматические тайминги forced-Viterbi (не сырые ручные)"
SELECTABLE = False
AUTO = False
ISOLATE = False
ALIGNED = True
PRIORITY = 90


def available() -> bool:
    return True


def ready(rec) -> bool:
    return False            # не авто-запускается


def run(rec, audio, quran, out_dir, stage=None):
    raise NotImplementedError("тестовый слот заполняется вручную (скриптом), не через run()")
