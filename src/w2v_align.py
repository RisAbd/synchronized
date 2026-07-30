"""wav2vec2 forced alignment — СВОЙ CTC-Viterbi поверх сырых эмиссий, БЕЗ whisperx.

Зачем НЕ whisperx (директива владельца 24.07: «Wave2Vec — отдельный, зачем ему WhisperX?»):
whisperx — лишь обёртка, что грузит ту же HF-модель wav2vec2 + свою align-рутину; её рутина на
мелодичном таджвиде СВАЛИВАЕТ серию слов в один момент (rec7: 6:102:10..103:7 все в 210.16с,
collapsed_words=14 → подсветка проскакивает). Здесь модель грузим напрямую через `transformers`,
эмиссии считаем сами (log-softmax логитов), выравниваем СВОИМ монотонным CTC-Viterbi — путь по
построению не «схлопывается» (каждой метке ≥1 кадр, времена строго растут), а мелодичную протяжку
(мадд) честно отдаём слову-держателю: слово владеет временем до онсета СЛЕДУЮЩЕГО слова.

Зачем вообще wav2vec2, а не MMS-forced (`falign.py`): MMS романизует арабский и СХЛОПЫВАЕТ огласовки
→ тянущейся гласной токена нет, конец слова садится на согласный спайк, протяжка падает в «дыру».
wav2vec2 (CTC поверх сырых арабских символов) держит слово сквозь мадд → границы честнее, coverage
считается по настоящим t_end.

Выдаёт sync_map ТОЙ ЖЕ формы, что `falign.align`: {meta, timeline, word_timeline, char_timeline},
совместимой с `player.build_data`. Вход: verses=[(surah, ayah, text), ...] (диапазон уже нашёл
`w2v_range` из СВОЕЙ акустики — БЕЗ ASR).

Модель + эмиссии — ТОЛЬКО GPU (правило проекта). Viterbi/снап — CPU (numpy). transformers/torch/
soundfile импортируются лениво (модуль импортируется где угодно; available() проверит deps).
"""
from __future__ import annotations

import os

import alignbricks     # общие кирпичи (_snap_bounds, _HARAKAT) — связь w2v→falign снята (директива 25.07)
import quran as quranmod

_HARAKAT = alignbricks._HARAKAT
SAMPLE_RATE = 16000

_MODEL_NAME = os.environ.get("SYNC_W2V_MODEL", "") or "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
_model = None
_processor = None
_vocab = None


def available() -> bool:
    """Есть ли transformers+torch+soundfile (без загрузки моделей)."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("transformers", "torch", "soundfile"))


def _norm(w: str) -> str:
    return quranmod.normalize(w)


def _load_model(device: str):
    """Загрузить HF-модель wav2vec2 напрямую (без whisperx). Кэшируется в процессе."""
    global _model, _processor, _vocab
    if _model is None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        _processor = Wav2Vec2Processor.from_pretrained(_MODEL_NAME)
        _model = Wav2Vec2ForCTC.from_pretrained(_MODEL_NAME).to(device).eval()
        _vocab = _processor.tokenizer.get_vocab()   # {символ: id}
    return _model, _processor, _vocab


def is_loaded() -> bool:
    return _model is not None


def warmup():
    """Прелоад модели (загрузка HF-весов на GPU ~5с). Зовём при подключении live-сессии в отдельном
    потоке, ПОКА клиент набирает первые ~6с буфера → загрузка прячется под неизбежный набор контекста,
    первый декод уже по тёплой модели (не морозит event-loop на 5с посреди стрима). No-op если загружена."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _load_model(device)
        return True
    except Exception:
        return False


def unload():
    """Выгрузить модель из процесса → освободить VRAM (~1.5ГБ). Для деинита live при простое
    (владелец 30.07: не держать видеопамять, когда никто не читает → транскрипция large-v3 получает
    память БЕЗ перезагрузки сервиса). Следующий emissions() лениво загрузит модель заново."""
    global _model, _processor, _vocab
    if _model is None:
        return False
    _model = None; _processor = None; _vocab = None
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return True


def _load_wav(path):
    """Аудио → float32 моно 16кГц (soundfile + librosa-ресемпл при нужде). Без whisperx/ffmpeg-CLI."""
    import numpy as np
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(wav, dtype="float32")


def emissions(audio_path, window_sec: float = 20.0):
    """wav2vec2 CTC log-softmax эмиссии по ВСЕЙ записи (окнами, чтобы влезть в 6ГБ) — GPU.

    Возвращает (E, stride_ms, idx2ch, ch2idx): E[кадр, класс] float32 (log-prob), шаг кадра в мс,
    словарь класс↔символ модели. Сырьё для независимого определения диапазона (`w2v_range`) и для
    детекта возвратов из СВОЕЙ акустики — БЕЗ данных других распознавателей. Вход нормализуется
    feature-экстрактором процессора (zero-mean/unit-var — как ждёт модель)."""
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("w2v emissions требует GPU")
    model, proc, vocab = _load_model(device)
    idx2ch = {int(v): k for k, v in vocab.items()}
    ch2idx = {k: int(v) for k, v in vocab.items()}

    audio = _load_wav(audio_path)
    # свёрточный стек wav2vec2 схлопывает ~320× → на аудио короче ~0.4с глубокая свёртка падает
    # (RuntimeError: kernel size can't be greater than input). В live первые чанки окна с микрофона
    # бывают крохотными → возвращаем None, вызыватель трактует как «пусто» (не крашим 500-й).
    _MIN_SAMPLES = int(0.4 * SAMPLE_RATE)
    if len(audio) < _MIN_SAMPLES:
        return None, 0.0, idx2ch, ch2idx
    dur = len(audio) / SAMPLE_RATE
    chunks, strides = [], []
    t = 0.0
    while t < dur:
        e = min(dur, t + window_sec)
        seg = audio[int(t * SAMPLE_RATE):int(e * SAMPLE_RATE)]
        if len(seg) < _MIN_SAMPLES:                 # хвостовой огрызок окна — пропускаем
            break
        iv = proc(seg, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
        with torch.inference_mode():
            emis = torch.log_softmax(model(iv).logits, dim=-1)[0].cpu().numpy().astype("float32")
        chunks.append(emis)
        strides.append((e - t) / len(emis))
        t = e
    torch.cuda.empty_cache()
    if not chunks:
        return None, 0.0, idx2ch, ch2idx
    E = np.concatenate(chunks, axis=0)
    stride_ms = float(np.mean(strides) * 1000)
    return E, stride_ms, idx2ch, ch2idx


def _ctc_viterbi(E, labels, blank: int):
    """Монотонный CTC-Viterbi: лучший путь, выравнивающий `labels` к кадрам E.

    Стандартный CTC: расширенная последовательность [blank, l0, blank, l1, ..., blank] (S=2L+1),
    переходы stay / +1 / +2 (скип blank между РАЗНЫМИ метками). Viterbi (max) с backpointer'ами,
    backtrack → path[t] = позиция в расширенной посл-ти на кадре t. Возвращает (path, ext).

    Ключевое: скип на blank-цель запрещён (ext[s]==ext[s-2]==blank) → путь ОБЯЗАН постоять на
    каждой метке ≥1 кадр → ни одно слово не «схлопывается» в ноль (в отличие от whisperx-рутины).

    Возвращает (path, ext, total_score), где total_score = лог-вероятность лучшего пути (alpha в
    конце). T один и тот же для двух проходов (одни эмиссии) → total_score прямо сравним между
    выравниваниями: дополненный повторами эталон, если он реально бьётся со вторым звучанием,
    даёт БОЛЬШЕ total_score (иначе — меньше, т.к. лишние метки тянутся по не-своим кадрам).
    """
    import numpy as np
    T = E.shape[0]
    L = len(labels)
    S = 2 * L + 1
    ext = np.empty(S, dtype=np.int64)
    ext[0] = blank
    ext[1::2] = labels
    ext[2::2] = blank
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    NEG = -1e30
    alpha = np.full(S, NEG, dtype=np.float64)
    alpha[0] = float(E[0, ext[0]])
    if S > 1:
        alpha[1] = float(E[0, ext[1]])
    bp = np.zeros((T, S), dtype=np.int8)     # 0=stay(s), 1=из s-1, 2=из s-2
    idxS = np.arange(S)
    for t in range(1, T):
        e_t = E[t, ext]                      # [S] gather (без материализации [T,S])
        frm1 = np.empty(S); frm1[0] = NEG; frm1[1:] = alpha[:-1]
        frm2 = np.full(S, NEG); frm2[2:] = np.where(skip[2:], alpha[:-2], NEG)
        cand = np.stack([alpha, frm1, frm2])  # [3,S]: stay / +1 / +2
        c = cand.argmax(axis=0)
        alpha = cand[c, idxS] + e_t
        bp[t] = c.astype(np.int8)

    end_s = S - 1 if (S == 1 or alpha[S - 1] >= alpha[S - 2]) else S - 2
    path = np.empty(T, dtype=np.int64)
    s = end_s
    for t in range(T - 1, -1, -1):
        path[t] = s
        cc = int(bp[t, s])
        if cc == 1:
            s -= 1
        elif cc == 2:
            s -= 2
    return path, ext, float(alpha[end_s])


def _ctc_viterbi_repeats(E, labels, lab_word, blank: int, R: int, P: float):
    """CTC-Viterbi С ВОЗВРАТАМИ (подход владельца: один проход, аллайнер сам ходит назад).

    К обычным переходам stay/+1/+2 добавлен ПРЫЖОК-НАЗАД: с последней буквы слова w путь может уйти
    на первую букву более раннего слова w' (w-R ≤ w' ≤ w) за штраф P. Возврат чтеца выражается
    нативно — путь идёт ...w, [jump] w', w'+1, ..., w, w+1... Прыжки ВПЕРЁД через слова структурно
    невозможны (только +1/+2 по буквам); время монотонно (кадры растут) → ложная петля «оплачивается»
    несоответствием акустики и не берётся. P — мягкий регуляризатор от микро-петель (по умолчанию 0:
    возврат бесплатен, акустика сама держит чистоту). Возвращает path[t] = позиция в ext.
    """
    import numpy as np
    T = int(E.shape[0]); L = len(labels)
    S = 2 * L + 1
    ext = np.empty(S, dtype=np.int64)
    ext[0] = blank; ext[1::2] = labels; ext[2::2] = blank
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    NEG = -1e30

    # старт/конец каждого слова в ext (по порядку слов, только слова С метками)
    lab_lo, lab_hi = {}, {}
    for li, gi in enumerate(lab_word):
        lab_lo.setdefault(gi, li); lab_hi[gi] = li
    words = sorted(lab_lo)
    starts = np.array([2 * lab_lo[w] + 1 for w in words])   # цель прыжка (первая буква)
    lasts = np.array([2 * lab_hi[w] + 1 for w in words])    # источник прыжка (последняя буква)
    nW = len(words)

    alpha = np.full(S, NEG); alpha[0] = float(E[0, ext[0]])
    if S > 1:
        alpha[1] = float(E[0, ext[1]])
    bp = np.full((T, S), -1, dtype=np.int32)     # предшественник-состояние
    idxS = np.arange(S)
    widx = np.arange(nW)
    for t in range(1, T):
        e_t = E[t, ext]
        frm1 = np.empty(S); frm1[0] = NEG; frm1[1:] = alpha[:-1]
        frm2 = np.full(S, NEG); frm2[2:] = np.where(skip[2:], alpha[:-2], NEG)
        cand = np.stack([alpha, frm1, frm2])
        c = cand.argmax(axis=0)
        best = cand[c, idxS]
        pred = np.where(c == 0, idxS, np.where(c == 1, idxS - 1, idxS - 2))
        # прыжок-назад: цель starts[w'], источник lasts[w], w' ≤ w ≤ w'+R
        last_vals = alpha[lasts]
        best_src_val = np.full(nW, NEG)
        best_src_w = np.zeros(nW, dtype=int)
        for off in range(0, R + 1):
            shifted = np.full(nW, NEG)
            if off == 0:
                shifted = last_vals.copy()
            else:
                shifted[:-off] = last_vals[off:]
            better = shifted > best_src_val
            best_src_val = np.where(better, shifted, best_src_val)
            best_src_w = np.where(better, np.minimum(widx + off, nW - 1), best_src_w)
        jump_val = best_src_val - P
        for k in np.where(jump_val > best[starts])[0]:
            s = int(starts[k])
            best[s] = jump_val[k]
            bp[t, s] = int(lasts[best_src_w[k]])
        alpha = best + e_t
        bp[t] = np.where(bp[t] >= 0, bp[t], pred)

    end_s = S - 1 if (S == 1 or alpha[S - 1] >= alpha[S - 2]) else S - 2
    path = np.empty(T, dtype=np.int64)
    s = end_s
    for t in range(T - 1, -1, -1):
        path[t] = s
        nxt = int(bp[t, s])
        s = nxt if nxt >= 0 else s
    return path


def forced_align(E, stride_ms: float, verses, idx2ch: dict, ch2idx: dict,
                 audio_path, snap: bool | None = None, slots=None) -> dict:
    """СВОЙ CTC-forced-align диапазона аятов к аудио по готовым эмиссиям E (Viterbi). GPU не нужен
    (эмиссии уже посчитаны в `emissions()`), считаем на CPU.

    verses — [(surah, ayah, text), ...] (диапазон нашёл w2v_range из своей акустики). Онсет каждого
    слова = время первого кадра его первой метки; слово ВЛАДЕЕТ временем до онсета следующего слова
    (мадд/хвост честно висит на слове-держателе), затем снап к тишине поджимает реальные паузы.

    slots — опциональный ЯВНЫЙ порядок слотов (list of (surah, ayah, wi, word, rep_bool)). Когда
    задан — эталон берём из него как есть (в т.ч. с ДУБЛИРОВАННЫМИ кусками-перечитками: тот же
    (surah,ayah,wi) встречается несколько раз подряд), и Viterbi монотонно раскладывает КАЖДОЕ
    звучание на свою копию → возврат выражается движением ВПЕРЁД по дублированному тексту, без
    latания пост-фактум (подход владельца: «виттербидуй сразу задублированный оригинал»). rep-слоты
    получают в word_timeline пометку rep=True. Когда slots=None — обычный эталон (каждое слово раз).
    """
    import numpy as np

    # выкинуть токены-вакфы/паузы из текста аятов — единая безвакфовая индексация wi (как build_data)
    verses = [(s, a, " ".join(quranmod.word_tokens(t))) for s, a, t in verses]

    # плоский ref: слоты по порядку чтения. rep_flags[i] = слот i — копия-перечитка.
    ref = []                     # (surah, ayah, wi, arabic_word)
    if slots is None:
        for surah, ayah, txt in verses:
            for wi, w in enumerate(txt.split()):
                ref.append((surah, ayah, wi, w))
        rep_flags = [False] * len(ref)
    else:
        rep_flags = []
        for surah, ayah, wi, w, rep in slots:
            ref.append((surah, ayah, wi, w))
            rep_flags.append(bool(rep))

    blank = ch2idx.get("<pad>", 0)
    labels, lab_word = [], []    # id-метки vocab + индекс слова (в ref) для каждой метки
    for gi, (_s, _a, _wi, w) in enumerate(ref):
        for ch in w:
            j = ch2idx.get(ch)
            if j is None or j == blank:
                continue         # символа нет в vocab (напр. надстрочный алеф U+0670) — пропускаем
            labels.append(j)
            lab_word.append(gi)

    T = int(E.shape[0])
    if not labels or T == 0:
        return _empty(ref)

    path, _ext, path_score = _ctc_viterbi(E, labels, blank)
    sec = stride_ms / 1000.0
    L = len(labels)

    # первый/последний кадр каждой метки (метка li в расширенной посл-ти на позиции 2*li+1 — нечётной)
    first = [-1] * L
    last = [-1] * L
    for t in range(T):
        s = int(path[t])
        if s & 1:                # нечётная позиция → реальная метка
            li = (s - 1) // 2
            if first[li] < 0:
                first[li] = t
            last[li] = t

    # онсет/конец слова из его меток
    w_first, w_last = {}, {}
    for li, gi in enumerate(lab_word):
        if first[li] < 0:
            continue
        if gi not in w_first:
            w_first[gi] = first[li]
        w_last[gi] = last[li]
    known = sorted(w_first)      # слова с метками, по возрастанию (путь монотонен → онсеты растут)

    bounds_opt = [None] * len(ref)
    for idx, gi in enumerate(known):
        f0 = w_first[gi]
        # слово владеет временем ДО онсета следующего слова (мадд/протяжка висит на держателе);
        # последнее слово — до последнего своего кадра (+1).
        f1 = w_first[known[idx + 1]] if idx + 1 < len(known) else (w_last[gi] + 1)
        t0 = f0 * sec
        t1 = max(f0 + 1, f1) * sec
        bounds_opt[gi] = (t0, t1)
    matched = len(known)

    bounds, interp_flags = _interp_missing(bounds_opt)

    # снап к тишине (RMS): поджать границы ТОЛЬКО внутрь к речи. Мадд = речь → держится; реальная
    # пауза → триммится (заливка замирает на 100%, подсветка ждёт след. слово). Опт-аут SYNC_W2V_SNAP=0.
    snapped = 0
    do_snap = (os.environ.get("SYNC_W2V_SNAP", "1") != "0") if snap is None else snap
    if do_snap:
        audio = _load_wav(audio_path)
        real_idx = [i for i, f in enumerate(interp_flags) if not f]
        real_bounds = [bounds[i] for i in real_idx]
        snapped_bounds, snapped = alignbricks._snap_bounds(real_bounds, audio)
        for k, i in enumerate(real_idx):
            bounds[i] = snapped_bounds[k]

    # сборка дорожек
    word_timeline, timeline, char_timeline = [], [], []
    seen_ayah = set()
    for i, (surah, ayah, wi, arabic) in enumerate(ref):
        t0, t1 = bounds[i]
        entry = {"t": round(t0, 3), "surah": surah, "ayah": ayah, "wi": wi}
        if not interp_flags[i] and t1 > t0:
            entry["t_end"] = round(t1, 3)
        if rep_flags[i]:
            entry["rep"] = True
        word_timeline.append(entry)
        if (surah, ayah) not in seen_ayah:
            seen_ayah.add((surah, ayah))
            timeline.append({"t": round(t0, 3), "surah": surah, "ayah": ayah})
        if t1 > t0:
            base_positions = [p for p, ch in enumerate(arabic) if ch not in _HARAKAT]
            nb = max(1, len(base_positions))
            for ci, ch in enumerate(arabic):
                kk = sum(1 for p in base_positions if p < ci)
                frac0 = kk / nb
                frac1 = (kk + 1) / nb
                ct0 = t0 + (t1 - t0) * frac0
                ct1 = t0 + (t1 - t0) * (frac0 if ch in _HARAKAT else frac1)
                char_timeline.append({"t": round(ct0, 3), "t_end": round(ct1, 3),
                                      "surah": surah, "ayah": ayah, "wi": wi, "ci": ci})

    # строгий рост t (страховка — онсеты уже растут) + чистка невалидного t_end
    for i in range(1, len(word_timeline)):
        if word_timeline[i]["t"] <= word_timeline[i - 1]["t"]:
            word_timeline[i]["t"] = round(word_timeline[i - 1]["t"] + 0.001, 3)
        te = word_timeline[i].get("t_end")
        if te is not None and te <= word_timeline[i]["t"]:
            del word_timeline[i]["t_end"]

    meta = {
        "aligner": "wav2vec2-ctc-viterbi",
        "align_model": _MODEL_NAME,
        "ref_words": len(ref),
        "aligned_units": matched,
        "coverage": round(matched / len(ref), 3) if ref else 0.0,
        "interpolated": sum(interp_flags),
        "snapped_to_silence": snapped,
        "wt": len(word_timeline),
        "ct": len(char_timeline),
        "path_score": round(path_score, 2),
        "reps": sum(1 for f in rep_flags if f),
        "device": "cuda",
    }
    return {"meta": meta, "timeline": timeline,
            "word_timeline": word_timeline, "char_timeline": char_timeline}


# ── Оракул правдоподобия: авто-структура повторов (WG, план владельца tg_4539) ────────────────
# Идея: forced_align(slots=) по РАСШИРЕННОМУ тексту повторов даёт почти идеал (эталон test2), но
# нужен сам расширенный текст. Генерим его сам: для фразы-кандидата в «переросшем» аяте сравниваем
# ЛОКАЛЬНЫЙ CTC path_score H0(фраза×m) vs H1(фраза×m+1) на кадрах окна вокруг неё; ΔH>margin ⟺
# лишняя копия реально легла на свои кадры (иначе метки тянутся по не-своим → score падает). Порогов
# «сколько повторов» нет — судит модель. Чистая функция от эмиссий (GPU/аудио не нужны).
_GREEDY_MARGIN = float(os.environ.get("SYNC_W2V_GREEDY_MARGIN", "3") or 3)   # мин. ΔH принятия копии
_GREEDY_MAXLEN = int(os.environ.get("SYNC_W2V_GREEDY_MAXLEN", "5") or 5)     # макс. длина фразы (слов)
_GREEDY_SPANK = float(os.environ.get("SYNC_W2V_GREEDY_SPANK", "2.8") or 2.8)  # порог спан/букву к медиане
_GREEDY_CTX = int(os.environ.get("SYNC_W2V_GREEDY_CTX", "2") or 2)           # контекст окна (слов)
_GREEDY_MAXMUL = int(os.environ.get("SYNC_W2V_GREEDY_MAXMUL", "3") or 3)     # макс. доп. копий фразы


def _labels_of(words, ch2idx, blank):
    lab = []
    for w in words:
        for ch in w:
            j = ch2idx.get(ch)
            if j is not None and j != blank:
                lab.append(j)
    return lab


def _canon_mono(E, canon, ch2idx):
    """Монотонная раскладка канона → (w_first, w_last, known) в КАДРАХ (для окон/спанов)."""
    blank = ch2idx.get("<pad>", 0)
    labels, lab_word = [], []
    for gi, c in enumerate(canon):
        for ch in c[3]:
            j = ch2idx.get(ch)
            if j is None or j == blank:
                continue
            labels.append(j)
            lab_word.append(gi)
    if not labels:
        return {}, {}, []
    path, _ext, _sc = _ctc_viterbi(E, labels, blank)
    T = int(E.shape[0])
    L = len(labels)
    first = [-1] * L
    last = [-1] * L
    for t in range(T):
        s = int(path[t])
        if s & 1:
            li = (s - 1) // 2
            if first[li] < 0:
                first[li] = t
            last[li] = t
    w_first, w_last = {}, {}
    for li, gi in enumerate(lab_word):
        if first[li] < 0:
            continue
        if gi not in w_first:
            w_first[gi] = first[li]
        w_last[gi] = last[li]
    return w_first, w_last, sorted(w_first)


def greedy_repeat_slots(E, verses, ch2idx, stride_ms, margin=None, maxlen=None,
                        span_k=None, ctx=None, maxmul=None, verbose=False):
    """Авто-структура повторов (WG). Возвращает slots [(surah, ayah, wi, word, rep_bool)] для
    forced_align(slots=). Порогов на число повторов нет — оракул правдоподобия судит каждую копию."""
    import statistics
    margin = _GREEDY_MARGIN if margin is None else margin
    maxlen = _GREEDY_MAXLEN if maxlen is None else maxlen
    span_k = _GREEDY_SPANK if span_k is None else span_k
    ctx = _GREEDY_CTX if ctx is None else ctx
    maxmul = _GREEDY_MAXMUL if maxmul is None else maxmul
    blank = ch2idx.get("<pad>", 0)
    sec = stride_ms / 1000.0

    verses = [(s, a, " ".join(quranmod.word_tokens(t))) for s, a, t in verses]
    canon = []
    for s, a, txt in verses:
        for wi, w in enumerate(txt.split()):
            canon.append((s, a, wi, w, False))
    if not canon:
        return canon

    w_first, w_last, known = _canon_mono(E, canon, ch2idx)
    if not known:
        return canon
    T = int(E.shape[0])

    def onset_frame(gi, default):
        for gj in range(gi, -1, -1):
            if gj in w_first:
                return w_first[gj]
        for gj in range(gi, len(canon)):
            if gj in w_first:
                return w_first[gj]
        return default

    # «переросшие» аяты — по АНОМАЛЬНО ДЛИННОМУ СЛОВУ: перечитка фразы всегда оставляет кадры лишнего
    # звучания, которые монотонный канон-Viterbi вынужден кому-то отдать → спан/букву у слова в зоне
    # повтора резко выше глобального темпа (rec7: ×5-8 против ×1-2 в спокойных). Гейт нужен для СКОРОСТИ
    # и локализации; истину внутри флаг-аята решает оракул (мадд-раздутое слово он отвергнет — ΔH<0).
    def _nbase(w):
        return max(1, len([c for c in w if c not in _HARAKAT]))

    word_rate = {}
    for idx, gi in enumerate(known):
        f0 = w_first[gi]
        f1 = w_first[known[idx + 1]] if idx + 1 < len(known) else (w_last[gi] + 1)
        word_rate[gi] = max(1, f1 - f0) / _nbase(canon[gi][3])
    med = statistics.median(word_rate.values()) if word_rate else 1.0

    ayah_gis = {}
    for gi in known:
        ayah_gis.setdefault((canon[gi][0], canon[gi][1]), []).append(gi)
    flagged = set()
    for (s, a), gis in ayah_gis.items():
        if med > 0 and any(word_rate[gi] / med > span_k for gi in gis):
            flagged.add((s, a))
    if verbose:
        print(f"переросшие аяты ({len(flagged)}/{len(ayah_gis)}): "
              f"{sorted(a for _, a in flagged)}; med={med:.1f} кадр/букву")

    by_ayah = {}
    for k, c in enumerate(canon):
        by_ayah.setdefault((c[0], c[1]), []).append(k)

    def dH_extra(i, j, extra):
        """ΔH добавления (extra+1)-й копии фразы canon[i..j] к extra уже имеющимся (сверх канона)."""
        lo = max(0, i - ctx)
        hi = min(len(canon) - 1, j + ctx)
        f0 = onset_frame(lo, 0)
        f1 = onset_frame(hi + 1, T) if hi + 1 < len(canon) else (w_last.get(hi, T - 1) + 1)
        f0 = max(0, min(int(f0), T - 1))
        f1 = max(f0 + 1, min(int(f1), T))
        Ew = E[f0:f1]
        pre = [canon[k][3] for k in range(lo, j + 1)]
        phrase = [canon[k][3] for k in range(i, j + 1)]
        post = [canon[k][3] for k in range(j + 1, hi + 1)]
        lab0 = _labels_of(pre + phrase * extra + post, ch2idx, blank)
        lab1 = _labels_of(pre + phrase * (extra + 1) + post, ch2idx, blank)
        if not lab0 or not lab1:
            return -1e30
        _, _, s0 = _ctc_viterbi(Ew, lab0, blank)
        _, _, s1 = _ctc_viterbi(Ew, lab1, blank)
        return s1 - s0

    # кандидаты-фразы в флаг-аятах; для каждой — ΔH первой доп-копии
    scored = []
    for (s, a) in flagged:
        idxs = by_ayah.get((s, a), [])
        for i in idxs:
            for j in idxs:
                if 0 <= j - i < maxlen:
                    d = dH_extra(i, j, 0)
                    if d > margin:
                        scored.append((d, i, j))
    scored.sort(reverse=True)   # больший ΔH раньше — жадно

    # принять без пересечений (одна канон-позиция участвует в ≤1 повторе); нарастить кратность
    used = set()
    accepted = []
    for d0, i, j in scored:
        span_set = set(range(i, j + 1))
        if span_set & used:
            continue
        extra = 1
        while extra < maxmul and dH_extra(i, j, extra) > margin:
            extra += 1
        used |= span_set
        accepted.append((i, j, extra, d0))
        if verbose:
            name = " ".join(canon[k][3] for k in range(i, j + 1))
            print(f"  +повтор [{canon[i][1]}:{canon[i][2]}-{canon[j][2]}] «{name}» ×{extra + 1} ΔH={d0:+.1f}")

    # вставить дубли после forward-вхождения фразы (по канон-позиции j); с конца — индексы не съезжают
    slots = list(canon)
    for i, j, extra, _d in sorted(accepted, key=lambda x: x[1], reverse=True):
        phrase = [(canon[k][0], canon[k][1], canon[k][2], canon[k][3], True) for k in range(i, j + 1)]
        keyj = (canon[j][0], canon[j][1], canon[j][2])
        pos = None
        for p, sl in enumerate(slots):
            if (sl[0], sl[1], sl[2]) == keyj and not sl[4]:
                pos = p
        if pos is None:
            continue
        ins = phrase * extra
        slots = slots[:pos + 1] + ins + slots[pos + 1:]
    return slots


_REPEAT_R = int(os.environ.get("SYNC_W2V_REPEAT_R", "9") or 9)
_REPEAT_P = float(os.environ.get("SYNC_W2V_REPEAT_P", "0") or 0)


def repeat_align(E, stride_ms: float, verses, idx2ch: dict, ch2idx: dict,
                 audio_path, snap: bool | None = None, R: int | None = None,
                 P: float | None = None) -> dict:
    """ОДИН проход CTC-Viterbi С ВОЗВРАТАМИ (подход владельца): аллайнер сам ходит назад по акустике.
    Возврат чтеца выражается как сегмент пути с УБЫВАЮЩИМ индексом слова → помечаем rep=True. Ни
    пред-детекта, ни дублирования эталона, ни жёстких порогов — только окно R и штраф P (по умолч. 0).
    Дорожки — В ПОРЯДКЕ ПУТИ (перечитка inline, движение вперёд между возвратами монотонно)."""
    import numpy as np
    R = _REPEAT_R if R is None else R
    P = _REPEAT_P if P is None else P

    verses = [(s, a, " ".join(quranmod.word_tokens(t))) for s, a, t in verses]
    ref = []
    for surah, ayah, txt in verses:
        for wi, w in enumerate(txt.split()):
            ref.append((surah, ayah, wi, w))
    blank = ch2idx.get("<pad>", 0)
    labels, lab_word = [], []
    for gi, (_s, _a, _wi, w) in enumerate(ref):
        for ch in w:
            j = ch2idx.get(ch)
            if j is None or j == blank:
                continue
            labels.append(j); lab_word.append(gi)
    T = int(E.shape[0])
    if not labels or T == 0:
        return _empty(ref)

    path = _ctc_viterbi_repeats(E, labels, lab_word, blank, R, P)
    sec = stride_ms / 1000.0

    # сегменты пути: непрерывные ряды одного слова → (gi, f0, f1)
    segs = []
    cur, f0 = None, 0
    for t in range(T):
        s = int(path[t])
        w = lab_word[(s - 1) // 2] if (s % 2 == 1) else None
        if w is not None and w != cur:
            if cur is not None:
                segs.append((cur, f0, t))
            cur, f0 = w, t
    if cur is not None:
        segs.append((cur, f0, T))

    # границы сегмента: онсет = f0; слово владеет временем до онсета СЛЕДУЮЩЕГО сегмента (мадд висит)
    seg_bounds = []
    for idx, (gi, a, b) in enumerate(segs):
        t0 = a * sec
        t1 = (segs[idx + 1][1] * sec) if idx + 1 < len(segs) else (b * sec)
        seg_bounds.append((gi, t0, max(t0 + 0.001, t1)))
    # rep-флаг сегмента: индекс слова МЕНЬШЕ максимума виденного (путь ушёл назад = возврат)
    maxw, rep_flags = -1, []
    for (gi, _t0, _t1) in seg_bounds:
        rep_flags.append(gi < maxw); maxw = max(maxw, gi)
    matched = len(set(gi for gi, _, _ in seg_bounds))

    # снап к тишине (RMS) на границах сегментов
    audio = _load_wav(audio_path)
    snapped = 0
    do_snap = (os.environ.get("SYNC_W2V_SNAP", "1") != "0") if snap is None else snap
    bounds = [(t0, t1) for (_gi, t0, t1) in seg_bounds]
    if do_snap and bounds:
        bounds, snapped = alignbricks._snap_bounds(bounds, audio)

    word_timeline, timeline, char_timeline = [], [], []
    seen_ayah = set()
    for idx, (gi, _t0, _t1) in enumerate(seg_bounds):
        surah, ayah, wi, arabic = ref[gi]
        t0, t1 = bounds[idx]
        entry = {"t": round(t0, 3), "surah": surah, "ayah": ayah, "wi": wi}
        if t1 > t0:
            entry["t_end"] = round(t1, 3)
        if rep_flags[idx]:
            entry["rep"] = True
        word_timeline.append(entry)
        if not rep_flags[idx] and (surah, ayah) not in seen_ayah:
            seen_ayah.add((surah, ayah))
            timeline.append({"t": round(t0, 3), "surah": surah, "ayah": ayah})
        if t1 > t0:
            base_positions = [p for p, ch in enumerate(arabic) if ch not in _HARAKAT]
            nb = max(1, len(base_positions))
            for ci, ch in enumerate(arabic):
                kk = sum(1 for p in base_positions if p < ci)
                frac0 = kk / nb
                ct0 = t0 + (t1 - t0) * frac0
                ct1 = t0 + (t1 - t0) * (frac0 if ch in _HARAKAT else (kk + 1) / nb)
                char_timeline.append({"t": round(ct0, 3), "t_end": round(ct1, 3),
                                      "surah": surah, "ayah": ayah, "wi": wi, "ci": ci})

    # добор канонических слов, ни разу не посещённых (без vocab-метки) — интерполяцией в forward-позиции
    present = {(e["surah"], e["ayah"], e["wi"]) for e in word_timeline if not e.get("rep")}
    for gi, (surah, ayah, wi, _w) in enumerate(ref):
        if (surah, ayah, wi) in present:
            continue
        # сосед слева (предыдущее канон. слово с временем) — берём его t как приближение
        tprev = None
        for gj in range(gi - 1, -1, -1):
            k = (ref[gj][0], ref[gj][1], ref[gj][2])
            e = next((x for x in word_timeline if not x.get("rep")
                      and (x["surah"], x["ayah"], x["wi"]) == k), None)
            if e:
                tprev = e["t"]; break
        word_timeline.append({"t": round((tprev if tprev is not None else 0.0) + 0.001, 3),
                              "surah": surah, "ayah": ayah, "wi": wi})

    word_timeline.sort(key=lambda e: e["t"])
    for i in range(1, len(word_timeline)):
        if word_timeline[i]["t"] <= word_timeline[i - 1]["t"]:
            word_timeline[i]["t"] = round(word_timeline[i - 1]["t"] + 0.001, 3)
        te = word_timeline[i].get("t_end")
        if te is not None and te <= word_timeline[i]["t"]:
            del word_timeline[i]["t_end"]

    n_rep = sum(1 for f in rep_flags if f)
    meta = {"aligner": "wav2vec2-ctc-viterbi-repeats", "align_model": _MODEL_NAME,
            "ref_words": len(ref), "aligned_units": matched,
            "coverage": round(matched / len(ref), 3) if ref else 0.0,
            "snapped_to_silence": snapped, "wt": len(word_timeline), "ct": len(char_timeline),
            "reps": n_rep, "repeat_R": R, "repeat_P": P, "device": "cuda"}
    return {"meta": meta, "timeline": timeline,
            "word_timeline": word_timeline, "char_timeline": char_timeline}


def _empty(ref) -> dict:
    """Вырожденный результат (нет меток/кадров) — пустые дорожки, чтобы пайплайн не падал."""
    return {"meta": {"aligner": "wav2vec2-ctc-viterbi", "ref_words": len(ref),
                     "aligned_units": 0, "coverage": 0.0, "wt": 0, "ct": 0},
            "timeline": [], "word_timeline": [], "char_timeline": []}


def _interp_missing(bounds):
    """Дыры (ref-слова без пары) — линейной интерполяцией между известными соседями.

    Интерполированным конца не даём (нулевая длина) — реального конца у них нет.
    Возвращает (bounds_full, interp_flags).
    """
    n = len(bounds)
    known = [i for i, b in enumerate(bounds) if b is not None]
    out = [(0.0, 0.0)] * n
    flags = [False] * n
    if not known:
        return out, flags
    for i in range(n):
        if bounds[i] is not None:
            out[i] = bounds[i]; continue
        flags[i] = True
        left = max([k for k in known if k < i], default=None)
        right = min([k for k in known if k > i], default=None)
        if left is not None and right is not None:
            t0 = bounds[left][1]; t1 = bounds[right][0]
            frac = (i - left) / (right - left)
            t = t0 + (t1 - t0) * frac
            out[i] = (t, t)
        elif left is not None:
            out[i] = (bounds[left][1], bounds[left][1])
        else:
            out[i] = (bounds[right][0], bounds[right][0])
    return out, flags
