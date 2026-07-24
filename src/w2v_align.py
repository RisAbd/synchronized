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

import falign          # только за хелперами (_snap_bounds, _HARAKAT); тяжёлые импорты у него ленивы
import quran as quranmod

_HARAKAT = falign._HARAKAT
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
    dur = len(audio) / SAMPLE_RATE
    chunks, strides = [], []
    t = 0.0
    while t < dur:
        e = min(dur, t + window_sec)
        seg = audio[int(t * SAMPLE_RATE):int(e * SAMPLE_RATE)]
        iv = proc(seg, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
        with torch.inference_mode():
            emis = torch.log_softmax(model(iv).logits, dim=-1)[0].cpu().numpy().astype("float32")
        chunks.append(emis)
        strides.append((e - t) / len(emis))
        t = e
    torch.cuda.empty_cache()
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
    return path, ext


def forced_align(E, stride_ms: float, verses, idx2ch: dict, ch2idx: dict,
                 audio_path, snap: bool | None = None) -> dict:
    """СВОЙ CTC-forced-align диапазона аятов к аудио по готовым эмиссиям E (Viterbi). GPU не нужен
    (эмиссии уже посчитаны в `emissions()`), считаем на CPU.

    verses — [(surah, ayah, text), ...] (диапазон нашёл w2v_range из своей акустики). Онсет каждого
    слова = время первого кадра его первой метки; слово ВЛАДЕЕТ временем до онсета следующего слова
    (мадд/хвост честно висит на слове-держателе), затем снап к тишине поджимает реальные паузы.
    """
    import numpy as np

    # выкинуть токены-вакфы/паузы из текста аятов — единая безвакфовая индексация wi (как build_data)
    verses = [(s, a, " ".join(quranmod.word_tokens(t))) for s, a, t in verses]

    # плоский ref: слова диапазона по порядку
    ref = []                     # (surah, ayah, wi, arabic_word)
    for surah, ayah, txt in verses:
        for wi, w in enumerate(txt.split()):
            ref.append((surah, ayah, wi, w))

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

    path, _ext = _ctc_viterbi(E, labels, blank)
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
    audio = _load_wav(audio_path)
    snapped = 0
    do_snap = (os.environ.get("SYNC_W2V_SNAP", "1") != "0") if snap is None else snap
    if do_snap:
        real_idx = [i for i, f in enumerate(interp_flags) if not f]
        real_bounds = [bounds[i] for i in real_idx]
        snapped_bounds, snapped = falign._snap_bounds(real_bounds, audio)
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
        "device": "cuda",
    }
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
