"""Мост между Django-сервисом и ядром конвейера в `src/`.

Гоняет: ingest (один раз на запись) → распознавание (whisper|google) → align → данные плеера.
Сырые ответы ASR и промежуточные выгрузки кладём по папкам записи, чтобы всё дебажилось:

    media/rec/<id>/audio.mp3
    media/rec/<id>/asr/<recognizer>/raw.json        — сырой ответ whisper/API как есть
    media/rec/<id>/asr/<recognizer>/transcript.json — нормализованный вход align (дебаг)
    media/rec/<id>/asr/<recognizer>/sync-map.json    — выход align (points/segments/timeline)

Ядро (`src/`) остаётся тонким и импортируемым; здесь только оркестрация под сервис.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

from django.conf import settings

# подключаем ядро пайплайна
if str(settings.PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(settings.PIPELINE_SRC))


# --- пути хранилища записи --------------------------------------------------

def rec_dir(rec_id: int) -> Path:
    return Path(settings.REC_DATA_DIR) / str(rec_id)


def run_dir(rec_id: int, recognizer: str) -> Path:
    return rec_dir(rec_id) / "asr" / recognizer


def _ensure_cudnn_path():
    """LD_LIBRARY_PATH на pip-путь cuDNN/cuBLAS (иначе faster-whisper падает)."""
    if "cudnn" in os.environ.get("LD_LIBRARY_PATH", ""):
        return
    import site
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        cudnn = Path(base) / "nvidia" / "cudnn" / "lib"
        cublas = Path(base) / "nvidia" / "cublas" / "lib"
        if cudnn.is_dir():
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn}:{cublas}:" + os.environ.get("LD_LIBRARY_PATH", "")
            return


@lru_cache(maxsize=1)
def _quran():
    from quran import Quran
    return Quran.load()


# --- шаги конвейера ---------------------------------------------------------

def ensure_audio(rec) -> Path:
    """Получить аудио записи ОДИН раз и положить в media/rec/<id>/audio.<ext>.
    Идемпотентно: если файл уже есть — просто возвращаем путь (в т.ч. legacy web/audio)."""
    d = rec_dir(rec.id)
    d.mkdir(parents=True, exist_ok=True)

    if rec.audio_filename:
        p = d / rec.audio_filename
        if p.is_file():
            _fill_meta(rec, p)
            return p
        legacy = Path(settings.AUDIO_DIR) / rec.audio_filename  # демо-записи
        if legacy.is_file():
            _fill_meta(rec, legacy)
            return legacy

    import ingest
    src = Path(ingest.fetch(rec.source_url, settings.WORK_DIR))
    audio_name = f"audio{src.suffix or '.mp3'}"
    dst = d / audio_name
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    rec.audio_filename = audio_name
    rec.save(update_fields=["audio_filename", "updated_at"])
    _fill_meta(rec, dst)
    return dst


def _ffprobe_duration(path: Path) -> float:
    """Длительность аудио в секундах через ffprobe (0.0 если не вышло)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30)
        return round(float(out.stdout.strip()), 1) if out.stdout.strip() else 0.0
    except Exception:
        return 0.0


def _audio_time_coverage(word_timeline, audio_duration, bin_sec: float = 10.0) -> float:
    """ЧЕСТНОЕ покрытие: доля ДЛИТЕЛЬНОСТИ АУДИО, реально покрытая размещёнными словами.

    Раньше `coverage` = aligned/asr_words (align.py) — самореферентно: распознаватель, услышавший
    6 слов на 20-минутной записи и разместивший все 6, получал 1.0. Здесь знаменатель — реальная
    длительность аудио (одна для всех прогонов записи → метрика сравнима между whisper/google/forced).

    Бьём аудио на бины по bin_sec и считаем долю бинов, в которых есть хоть одно слово. Не обмануть
    ни малым числом слов (6 слов в первых 20с из 1295 → ~1-2%), ни двумя словами по краям (это дало бы
    полный span, но пустые бины в середине → низкое покрытие). Требует слов, РАЗМАЗАННЫХ по всей записи.
    """
    if not word_timeline or not audio_duration or audio_duration <= 0:
        return 0.0
    nbins = max(1, int(audio_duration // bin_sec) + (1 if audio_duration % bin_sec else 0))
    hit = set()
    for w in word_timeline:
        t = w.get("t")
        if t is None:
            continue
        b = int(t // bin_sec)
        if 0 <= b < nbins:
            hit.add(b)
    return round(len(hit) / nbins, 3)


# нормальная длительность слова, когда у точки нет t_end (таймлайны align.py — только t):
# берём зазор до следующего слова, но не больше этого потолка (иначе пауза «засчиталась» бы речью).
_WORD_SPAN_CAP = 0.6


def _speech_time_coverage(word_timeline, audio_duration) -> tuple[float, float]:
    """ТОЧНОЕ покрытие речью: объединение интервалов слов [t, t_end] / длительность аудио.

    Точнее 10-секундных бинов `_audio_time_coverage`: меряем реальные секунды, где размещено
    слово (а не «бин, куда попало хоть одно»). Отвечает на вопрос владельца «сколько % времени
    видео со словами, сколько без» (без = 1 − доля). Знаменатель — реальная длительность аудио
    (одна для всех прогонов → метрика сравнима между google/whisper/forced), поэтому по-прежнему
    штрафует и «6 слов на 20 мин», и «два слова по краям» (объединение = крохи → доля ~0).

    Возвращает (секунды_со_словами, доля[0..1]). t_end есть у forced (акустика) и у ЯКОРНЫХ
    слов google/whisper (реальный конец от распознавателя, align протаскивает его в word_timeline).
    Только у интерполированных между якорями слов t_end нет → длительность приближаем зазором до
    следующего с потолком _WORD_SPAN_CAP (иначе пауза «засчиталась» бы речью).
    """
    if not word_timeline or not audio_duration or audio_duration <= 0:
        return 0.0, 0.0
    pts = sorted((w for w in word_timeline if w.get("t") is not None), key=lambda w: w["t"])
    ivs = []
    for i, w in enumerate(pts):
        t0 = float(w["t"])
        te = w.get("t_end")
        if te is not None and float(te) > t0:
            t1 = float(te)
        else:  # нет t_end → зазор до следующего слова, но не больше потолка
            nxt = pts[i + 1]["t"] if i + 1 < len(pts) else t0 + _WORD_SPAN_CAP
            t1 = t0 + min(_WORD_SPAN_CAP, max(0.0, nxt - t0)) if nxt > t0 else t0 + _WORD_SPAN_CAP
        t0 = max(0.0, min(t0, audio_duration))
        t1 = max(t0, min(t1, audio_duration))
        ivs.append((t0, t1))
    # слияние перекрытий
    ivs.sort()
    covered, cs, ce = 0.0, None, None
    for a, b in ivs:
        if cs is None:
            cs, ce = a, b
        elif a <= ce:
            ce = max(ce, b)
        else:
            covered += ce - cs
            cs, ce = a, b
    if cs is not None:
        covered += ce - cs
    return round(covered, 1), round(covered / audio_duration, 3)


def _yt_title(url: str) -> str:
    """Название YouTube-ролика через публичный oEmbed (без ключа/зависимостей). '' при неудаче."""
    try:
        import urllib.parse
        import urllib.request
        api = "https://www.youtube.com/oembed?" + urllib.parse.urlencode(
            {"url": url, "format": "json"})
        with urllib.request.urlopen(api, timeout=8) as r:
            return (json.loads(r.read().decode()) or {}).get("title", "") or ""
    except Exception:
        return ""


def _fill_meta(rec, path: Path) -> None:
    """Заполнить rec.meta метаинфой источника (П6): длительность/размер/расширение/превью/название.
    Идемпотентно и без падений — метаинфо не критично для конвейера. Для YouTube название и превью
    берём без нового API (oEmbed + img.youtube.com). Если названия у записи нет — подставляем из меты."""
    meta = dict(rec.meta or {})
    fields = ["meta"]
    try:
        st = path.stat()
        meta.setdefault("ext", path.suffix)
        meta["filesize"] = st.st_size
        if not meta.get("duration"):
            dur = _ffprobe_duration(path)
            if dur:
                meta["duration"] = dur
    except OSError:
        pass

    if rec.youtube_id:
        meta.setdefault("thumbnail", f"https://img.youtube.com/vi/{rec.youtube_id}/hqdefault.jpg")
        if not meta.get("yt_title"):
            t = _yt_title(rec.source_url)
            if t:
                meta["yt_title"] = t
                if not rec.title:               # пустое название → подставим из YouTube
                    rec.title = t[:300]
                    fields.append("title")

    rec.meta = meta
    fields.append("updated_at")
    rec.save(update_fields=fields)


def _recognize(audio_path: Path, recognizer: str, rec, out: Path):
    """Распознать аудио выбранным бэкендом, СОХРАНИТЬ сырой ответ рядом (raw.json),
    вернуть нормализованные слова для align. Новый распознаватель = ветка здесь + запись
    в recognizers.REGISTRY."""
    import align as align_mod
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.json"
    tr_path = out / "transcript.json"

    if recognizer == "google":
        # 1) уже распознавали этот прогон (raw.json есть) — переиспользуем, чтобы не жечь квоту.
        # 2) иначе кэш ответов из старого проекта (ключ = gstt_key записи или stem аудио) — бесплатно.
        # 3) иначе живой Google STT API (если задан ключ+бакет) — сохраняем ответ в raw.json.
        if raw_path.is_file() and "results" in json.loads(raw_path.read_text() or "{}"):
            src = raw_path
        else:
            key = rec.gstt_key or audio_path.stem
            cache = Path(settings.GSTT_CACHE_DIR) / key / "gstt_response.json"
            if cache.is_file():
                raw = json.loads(cache.read_text())
                raw_path.write_text(json.dumps(raw, ensure_ascii=False))
            else:
                import gstt
                if not (settings.GSTT_LIVE and gstt.is_available()):
                    raise FileNotFoundError(
                        f"нет кэша Google STT для '{key}' ({cache}), а живой API выключен "
                        f"(нужны env GOOGLE_APPLICATION_CREDENTIALS + SYNC_GSTT_BUCKET, "
                        f"SYNC_GSTT_LIVE≠0).")
                resp = gstt.recognize(audio_path, bucket_name=settings.GSTT_BUCKET)
                raw_path.write_text(json.dumps(resp, ensure_ascii=False, indent=2))
            src = raw_path
        words = align_mod.load_transcript(src)
    elif recognizer == "whisper":
        # уже распознавали (raw.json со словами есть) — переиспользуем, НЕ жжём GPU заново
        # (симметрично google выше; сырой ответ на то и сохраняем). Иначе — живой whisper.
        if not (raw_path.is_file() and json.loads(raw_path.read_text() or "{}").get("words")):
            _ensure_cudnn_path()
            import asr
            raw = asr.transcribe(str(audio_path), language="ar")
            raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
        words = align_mod.load_transcript(raw_path)
    else:
        raise ValueError(f"неизвестный распознаватель: {recognizer!r}")

    # нормализованный вход align — для дебага (что реально скормили аллайнеру)
    tr_path.write_text(json.dumps(
        [{"word": w.word, "start": w.start, "end": w.end, "norm": w.norm} for w in words],
        ensure_ascii=False, indent=2))
    return words


def _forced_source(rec):
    """Готовый прогон-источник для forced align: из него берём диапазон читаемых аятов.
    Лучший по честному покрытию времени аудио (metrics.coverage), при равенстве —
    google > whisper. Фикс-приоритет google отдавал forced мусор, когда google плох:
    rec10 google cov 0.591 с ложным диапазоном 16:98 против whisper 0.803 (реальные 55:1→56)."""
    from .models import AsrRun
    from . import recognizers as rz
    ready = [r for r in rec.runs.all()
             if r.status == AsrRun.Status.READY and r.data and not rz.is_aligner(r.recognizer)]
    prio = {"google": 0, "whisper": 1}
    return max(ready, key=lambda r: ((r.metrics or {}).get("coverage") or 0.0,
                                     -prio.get(r.recognizer, 9)), default=None)


def _run_aligner_subprocess(rec_id: int, recognizer: str, out: Path) -> None:
    """Запустить выравниватель (forced/w2v) отдельным процессом `python -m recitations.gpu_align`.

    GPU-изоляция на 6ГБ-карте (см. gpu_align): подпроцесс освобождает VRAM целиком на выходе.
    Бросает RuntimeError с хвостом stderr при ненулевом коде возврата или отсутствии sync-map.json.
    Окружение (PYTHONNOUSERSITE/HOME/HF_HOME/NLTK_DATA/LD_LIBRARY_PATH/SYNC_*) наследуется."""
    cmd = [sys.executable, "-m", "recitations.gpu_align", str(rec_id), recognizer, str(out)]
    proc = subprocess.run(cmd, cwd=str(settings.BASE_DIR), capture_output=True, text=True)
    if proc.returncode != 0 or not (out / "sync-map.json").exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError(
            f"выравнивание ({recognizer}) в подпроцессе упало (код {proc.returncode}): "
            + " / ".join(tail))


def run_one(run, on_stage=None) -> None:
    """Прогнать конвейер одним распознавателем/выравнивателем для прогона AsrRun.
    Мутирует/сохраняет run. Аудио должно быть уже получено (ensure_audio). Бросает исключение
    при ошибке — статус/ошибку ведёт вызывающий (tasks)."""
    from .models import AsrRun
    from . import recognizers as rz

    rec = run.recitation

    def stage(name):
        run.stage = name
        run.save(update_fields=["stage", "updated_at"])
        if on_stage:
            on_stage(name)

    from player import build_data

    q = _quran()
    run.status = AsrRun.Status.PROCESSING
    run.error = ""
    run.save(update_fields=["status", "error", "updated_at"])
    t0 = time.monotonic()

    audio = ensure_audio(rec)
    out = run_dir(rec.id, run.recognizer)

    if rz.is_aligner(run.recognizer):
        # выравнивание: НЕ распознаём, а выравниваем известный текст аятов к аудио.
        # диапазон читаемого берём из готового ASR-прогона (align.py уже определил, что читается).
        import falign
        src = _forced_source(rec)
        if src is None:
            raise RuntimeError(
                "нет готового прогона (google/whisper) для диапазона аятов — сначала распознайте "
                "запись каким-нибудь ASR, затем добавьте выравнивание")
        verses = falign.verses_from_data(src.data)
        if not verses:
            raise RuntimeError(f"в прогоне-источнике '{src.recognizer}' нет разделов/аятов")
        stage("align")
        out.mkdir(parents=True, exist_ok=True)
        # Выравнивание (forced MMS / w2v) — в ОТДЕЛЬНОМ процессе (GPU-изоляция, см. gpu_align):
        # onnxruntime-forced держит липкую CUDA-арену, torch-w2v в том же процессе → OOM на 6ГБ.
        # Подпроцесс грузит фреймворк, пишет sync-map.json, выходит → VRAM освобождается целиком.
        _run_aligner_subprocess(rec.id, run.recognizer, out)
        sync_map = json.loads((out / "sync-map.json").read_text())
    else:
        import align as align_mod
        stage("asr")
        words = _recognize(audio, run.recognizer, rec, out)
        stage("align")
        sync_map = align_mod.align(words, q)
        # счётчики ASR↔эталон (идея quran-align): hits/subs/ins/dels/wer против текста
        # найденного диапазона — объективная «каша» распознавания, попадёт в run.metrics
        sync_map.setdefault("meta", {})["match"] = align_mod.match_stats(
            [w.norm for w in words], sync_map, q)
        (out / "sync-map.json").write_text(json.dumps(sync_map, ensure_ascii=False, indent=2))

    stage("build")
    data = build_data(sync_map, q, rec.audio_filename)
    # РЕАЛЬНАЯ длительность аудио (не последняя точка таймлайна — она самореферентна: у 6-словного
    # whisper timeline кончается на ~20с, хотя запись 1295с). Берём из meta (ffprobe при ingest),
    # иначе перепробуем ffprobe. Нужна как честный знаменатель покрытия.
    audio_dur = (rec.meta or {}).get("duration") or _ffprobe_duration(audio)
    tl_end = data["timeline"][-1]["t"] if data.get("timeline") else 0
    data["duration"] = round(audio_dur or tl_end)
    # посимвольная дорожка (forced align) — build_data её не копирует, тащим для побуквенной подсветки
    if sync_map.get("char_timeline"):
        data["char_timeline"] = sync_map["char_timeline"]

    wt = data.get("word_timeline") or []
    tl = data.get("timeline") or []
    meta = dict(sync_map.get("meta", {}))
    # старое coverage движка (aligned/asr_words или n/ref) — самореферентно, НЕ headline. Сохраняем
    # под ясным именем для дебага, а headline coverage считаем честно по времени аудио.
    if "coverage" in meta:
        meta["aligned_ratio"] = meta.pop("coverage")
    dur_for_cov = audio_dur or tl_end
    speech_sec, speech_ratio = _speech_time_coverage(wt, dur_for_cov)   # точная (объединение слов)
    bins_cov = _audio_time_coverage(wt, dur_for_cov)                    # грубая (10с-бины) — дебаг
    run.data = data
    run.metrics = {**meta,
                   "coverage": speech_ratio,           # headline: доля ВРЕМЕНИ со словами (точная)
                   "speech_sec": speech_sec,           # секунд со словами
                   "silence_sec": round(max(0.0, (dur_for_cov or 0) - speech_sec), 1),  # без слов
                   "coverage_bins": bins_cov,          # старая грубая метрика (сравнение)
                   "wt": len(wt), "tl": len(tl), "duration": round(audio_dur or tl_end),
                   "elapsed_sec": round(time.monotonic() - t0, 1)}
    run.status = AsrRun.Status.READY
    run.stage = ""
    run.save(update_fields=["data", "metrics", "status", "stage", "updated_at"])

    if not rec.title_ar and data.get("sections"):
        rec.title_ar = data["sections"][0]["title"]
        rec.save(update_fields=["title_ar", "updated_at"])


def build_manual_run(run, word_timeline: list[dict]) -> None:
    """Сохранить ручную привязку (П12 v2) как готовый прогон-выравниватель «manual».

    `word_timeline` — точки [{surah,ayah,wi,t,t_end}] из ручного элайнера (индексы wi — канон
    Tanzil, как у всех прогонов). Собираем `sync_map` (аятные якоря + слова) и прогоняем через
    ТОТ ЖЕ `build_data`, что forced/ASR → прогон получает единый формат data (sections/timeline/
    word_timeline) и становится выбираемым в плеере наравне с остальными. Синхронно (быстро, без
    нейросети) — зовётся прямо из вьюхи. Мутирует/сохраняет run."""
    from .models import AsrRun
    from player import build_data

    rec = run.recitation
    q = _quran()

    # нормализуем/валидируем точки (координаты из браузера — не доверяем вслепую)
    wt: list[dict] = []
    for w in word_timeline or []:
        try:
            s, a, wi, t = int(w["surah"]), int(w["ayah"]), int(w["wi"]), float(w["t"])
        except (KeyError, TypeError, ValueError):
            continue
        item = {"surah": s, "ayah": a, "wi": wi, "t": round(t, 3)}
        if w.get("t_end") is not None:
            try:
                item["t_end"] = round(float(w["t_end"]), 3)
            except (TypeError, ValueError):
                pass
        wt.append(item)
    wt.sort(key=lambda w: w["t"])
    if not wt:
        raise ValueError("пустой word_timeline — нечего сохранять")

    # аятные якоря для build_data: одна точка на (surah,ayah) в самое раннее t, по возрастанию t
    first_t: dict[tuple[int, int], float] = {}
    for w in wt:
        key = (w["surah"], w["ayah"])
        if key not in first_t or w["t"] < first_t[key]:
            first_t[key] = w["t"]
    timeline = [{"t": t, "surah": s, "ayah": a}
                for (s, a), t in sorted(first_t.items(), key=lambda kv: kv[1])]

    sync_map = {"timeline": timeline, "word_timeline": wt, "meta": {"source": "manual"}}

    run.status = AsrRun.Status.PROCESSING
    run.error = ""
    run.stage = "build"
    run.save(update_fields=["status", "error", "stage", "updated_at"])

    data = build_data(sync_map, q, rec.audio_filename)
    audio = rec_dir(rec.id) / (rec.audio_filename or "")
    audio_dur = (rec.meta or {}).get("duration") or (_ffprobe_duration(audio) if audio.is_file() else 0)
    tl_end = data["timeline"][-1]["t"] if data.get("timeline") else 0
    data["duration"] = round(audio_dur or tl_end or (wt[-1].get("t_end") or wt[-1]["t"]))

    dur_for_cov = audio_dur or data["duration"]
    speech_sec, speech_ratio = _speech_time_coverage(wt, dur_for_cov)
    bins_cov = _audio_time_coverage(wt, dur_for_cov)
    run.data = data
    run.metrics = {"source": "manual",
                   "coverage": speech_ratio,
                   "speech_sec": speech_sec,
                   "silence_sec": round(max(0.0, (dur_for_cov or 0) - speech_sec), 1),
                   "coverage_bins": bins_cov,
                   "wt": len(wt), "tl": len(timeline),
                   "duration": data["duration"]}
    run.status = AsrRun.Status.READY
    run.stage = ""
    run.save(update_fields=["data", "metrics", "status", "stage", "updated_at"])

    if not rec.title_ar and data.get("sections"):
        rec.title_ar = data["sections"][0]["title"]
        rec.save(update_fields=["title_ar", "updated_at"])
