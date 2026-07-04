#!/usr/bin/env python3
"""uxcheck — «глаза и сверка синхрона» для плеера, чтобы Claude мог сам тестировать фронт.

Claude не видит экран и не слышит звук. Этот харнесс даёт и то и другое СУРРОГАТНО:
  * ГЛАЗА  — headless Chrome (playwright, системный google-chrome) снимает плеер на
             нескольких вьюпортах (телефон/планшет/десктоп/большой). Скрины Claude
             открывает Read-ом и буквально видит: где уползло, где тесно, читаема ли харака,
             стоит ли активное слово на нужной ~40%-линии.
  * СИНХРОН — слышать не нужно. Проверяем ДЕТЕРМИНИРОВАННО по данным:
             1) рендер-корректность: для набора моментов t зовём update(t) в странице и
                сверяем, что подсветилось (.hot) именно то слово, что даёт bsearch(WT,t),
                и что элемент для него вообще построен (ловит дыры в word_timeline/DOM);
             2) здоровье таймлайна (по data.json, без браузера): покрытие, строгий рост t,
                разрывы, темп слов/сек, «скомканное начало» (симптом фантомного аята —
                куча слов в первые секунды), первый/последний t против длительности.

Запуск (web поднят в docker-compose на :8000):
    python3 tools/uxcheck.py                     # записи 5 6 7, все вьюпорты
    python3 tools/uxcheck.py --rec 5             # одна запись
    python3 tools/uxcheck.py --rec 5 --asr google  # конкретный прогон
    python3 tools/uxcheck.py --shots-only        # только скрины (быстро)

Скрины кладём в scratchpad (не в репо): --out меняет папку. Печатаем текстовый отчёт +
строку SUMMARY-JSON для машинного разбора.

Зависимости: playwright (pip, --user), системный google-chrome. Запускать на ХОСТЕ.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

# именованные вьюпорты: (w, h, device_scale). Телефоны — retina-скейл, чтобы харака читалась.
VIEWPORTS = {
    "s-phone": (360, 800, 2),    # тесный Android — сюда чаще всего «не влезает»
    "phone":   (390, 844, 2),    # типичный iPhone
    "tablet":  (768, 1024, 2),
    "desktop": (1440, 900, 1),
    "large":   (1920, 1080, 1),  # тут раньше активное слово липло к верху
}
DEFAULT_ORDER = ["s-phone", "phone", "tablet", "desktop", "large"]

# JS: собрать отчёт по вёрстке (оверфлоу, обрезки, элементы за краем)
JS_LAYOUT = """() => {
  const iw = window.innerWidth, root = document.documentElement;
  const off = [...document.querySelectorAll('body *')]
    .filter(e => e.getBoundingClientRect().right > iw + 1 && e.offsetParent !== null)
    .slice(0, 10)
    .map(e => ({tag:e.tagName.toLowerCase(), cls:e.className, right:Math.round(e.getBoundingClientRect().right)}));
  return {
    innerW: iw, innerH: window.innerHeight,
    scrollW: root.scrollWidth, scrollH: root.scrollHeight,
    hOverflow: root.scrollWidth > iw + 1,
    offenders: off,
    header: Math.round(document.querySelector('header').offsetHeight),
    words: document.querySelectorAll('.w').length,
  };
}"""

# JS: перемотать на t, свести подсветку, доехать глайдом до целевой линии, вернуть замер
JS_SEEK_MEASURE = """(t) => {
  update(t);
  // симулируем сходимость глайда (в норме это делает rAF): двигаем скролл к цели
  for (let k=0; k<600; k++) glide();
  update(t);
  let hot=null, hotRect=null;
  for (const key in wordEl) {
    if (wordEl[key].classList.contains('hot')) {
      hot = key; const r = wordEl[key].getBoundingClientRect();
      hotRect = {top: Math.round(r.top), bottom: Math.round(r.bottom), left: Math.round(r.left), right: Math.round(r.right)};
      break;
    }
  }
  const i = bsearch(WT, t);
  const exp = i < 0 ? null : (WT[i].surah+':'+WT[i].ayah+':'+WT[i].wi);
  const header = document.querySelector('header').offsetHeight;
  const desired = header + (window.innerHeight - header) * 0.40;
  return {
    hot, expected: exp,
    match: hot === exp,
    builtForExpected: exp ? (wordEl[exp] ? true : false) : null,
    hotRect, desired: Math.round(desired), innerH: window.innerHeight,
    deltaFromTarget: hotRect ? Math.round(hotRect.top - desired) : null,
    visible: hotRect ? (hotRect.top >= 0 && hotRect.bottom <= window.innerHeight) : false,
    badge: (document.getElementById('badgeAyah')||{}).textContent || '',
  };
}"""


def fetch_data(base: str, rec: int, asr: str | None) -> dict:
    url = f"{base}/r/{rec}/data.json" + (f"?asr={asr}" if asr else "")
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode())


def analyze_timeline(data: dict) -> dict:
    """Здоровье синхрона по данным (без браузера)."""
    wt = data.get("word_timeline") or []
    dur = data.get("duration") or (wt[-1]["t"] if wt else 0)
    total_words = sum(len(v.get("words") or v["text"].split())
                      for sec in data.get("sections", []) for v in sec.get("ayat", []))
    flags = []
    if not wt:
        return {"words_wt": 0, "flags": ["нет word_timeline"]}

    # строгий рост t
    non_mono = sum(1 for a, b in zip(wt, wt[1:]) if b["t"] < a["t"])
    if non_mono:
        flags.append(f"немонотонных t: {non_mono}")

    # разрывы между словами
    dts = [b["t"] - a["t"] for a, b in zip(wt, wt[1:])]
    max_gap = max(dts) if dts else 0
    if max_gap > 6:
        flags.append(f"большой разрыв {max_gap:.1f}s между соседними словами")

    # темп: слова/сек по секундным корзинам; всплеск = скомканный участок
    t0, t1 = wt[0]["t"], wt[-1]["t"]
    span = max(0.001, t1 - t0)
    pace = len(wt) / span
    # «скомканное начало» — симптом фантомного ведущего аята (баг rec5)
    head = [w for w in wt if w["t"] <= t0 + 2.0]
    head_pace = len(head) / 2.0
    if head_pace > 4:
        flags.append(f"скомканное начало: {len(head)} слов в первые 2s (симптом фантомного аята)")

    # покрытие: сколько слов текста получили тайминг
    coverage = round(len(wt) / total_words, 3) if total_words else 0
    if coverage < 0.95:
        flags.append(f"низкое покрытие таймингом: {coverage} ({len(wt)}/{total_words})")

    # хвост против длительности
    if dur and t1 < dur * 0.9:
        flags.append(f"последнее слово на {t1:.1f}s при длительности {dur}s (обрывается рано?)")

    return {
        "words_wt": len(wt), "words_text": total_words, "coverage": coverage,
        "duration": dur, "first_t": round(t0, 2), "last_t": round(t1, 2),
        "pace_wps": round(pace, 2), "head_words_2s": len(head),
        "max_gap_s": round(max_gap, 2), "non_monotonic": non_mono,
        "aligner": (data.get("meta") or {}).get("aligner") or "?",
        "flags": flags,
    }


def sample_times(data: dict, n: int) -> list[float]:
    wt = data.get("word_timeline") or []
    if not wt:
        return []
    t0, t1 = wt[0]["t"], wt[-1]["t"]
    if n <= 1:
        return [(t0 + t1) / 2]
    return [round(t0 + (t1 - t0) * k / (n - 1), 2) for k in range(n)]


def run(base: str, recs: list[int], asr: str | None, viewports: list[str],
        samples: int, out: Path, shots_only: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary = {"recs": {}}

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        for rec in recs:
            print(f"\n{'='*70}\n[rec {rec}] {base}/r/{rec}/" + (f"?asr={asr}" if asr else ""))
            data = fetch_data(base, rec, asr)

            # --- здоровье синхрона (по данным) ---
            th = analyze_timeline(data)
            print(f"  aligner={th.get('aligner')}  слов(wt)={th.get('words_wt')}/{th.get('words_text')} "
                  f"coverage={th.get('coverage')}  дл={th.get('duration')}s "
                  f"[{th.get('first_t')}..{th.get('last_t')}]s  темп={th.get('pace_wps')} сл/с")
            if th.get("flags"):
                for f in th["flags"]:
                    print(f"    ⚠ {f}")
            else:
                print("    ✓ таймлайн выглядит здоровым")

            times = sample_times(data, samples)
            rec_sum = {"timeline": th, "viewports": {}}

            for vp in viewports:
                w, h, scale = VIEWPORTS[vp]
                page = browser.new_page(viewport={"width": w, "height": h},
                                        device_scale_factor=scale)
                page.goto(f"{base}/r/{rec}/" + (f"?asr={asr}" if asr else ""),
                          wait_until="networkidle")
                page.wait_for_function("document.querySelectorAll('.w').length>0", timeout=20000)
                page.wait_for_timeout(400)  # шрифты/лейаут устаканиться

                layout = page.evaluate(JS_LAYOUT)
                vp_sum = {"layout": layout}
                tag = []
                if layout["hOverflow"]:
                    tag.append(f"h-оверфлоу scrollW={layout['scrollW']}>iw={layout['innerW']}")
                    off = layout["offenders"][:3]
                    tag.append("виновники: " + ", ".join(f"{o['tag']}.{o['cls']}@{o['right']}" for o in off))

                # money-shot: середина чтения, слово сведено к 40%-линии
                mid = times[len(times)//2] if times else 5
                m = page.evaluate(JS_SEEK_MEASURE, mid)
                shot = out / f"rec{rec}_{vp}_mid.png"
                page.screenshot(path=str(shot))  # только вьюпорт (что видит юзер)
                vp_sum["mid"] = m
                vp_sum["shot"] = str(shot)
                if m["deltaFromTarget"] is not None and abs(m["deltaFromTarget"]) > 60:
                    tag.append(f"слово не на 40%-линии: Δ={m['deltaFromTarget']}px (цель {m['desired']})")
                if m["match"] is False:
                    tag.append(f"РАССИНХРОН рендера: hot={m['hot']} ожид={m['expected']}")

                # --- сверка синхрона по выборке моментов (только если не shots-only) ---
                if not shots_only and times:
                    mism, notbuilt = [], []
                    for t in times:
                        r = page.evaluate(JS_SEEK_MEASURE, t)
                        if r["match"] is False:
                            mism.append((t, r["hot"], r["expected"]))
                        if r["builtForExpected"] is False:
                            notbuilt.append((t, r["expected"]))
                    vp_sum["sync_mismatch"] = mism
                    vp_sum["sync_notbuilt"] = notbuilt
                    if mism:
                        tag.append(f"рассинхрон рендера в {len(mism)}/{len(times)} точках")
                    if notbuilt:
                        tag.append(f"нет DOM-слова для {len(notbuilt)} ожидаемых ключей")

                status = "  ".join(tag) if tag else "✓ ок"
                print(f"    [{vp:8} {w}x{h}] {status}   → {shot.name}")
                rec_sum["viewports"][vp] = vp_sum
                page.close()

            summary["recs"][str(rec)] = rec_sum
        browser.close()

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nСкрины и summary.json → {out}")
    print("SUMMARY-JSON " + json.dumps({
        r: {"flags": s["timeline"].get("flags", []),
            "vp_issues": {vp: (v.get("sync_mismatch") and len(v["sync_mismatch"]) or 0)
                          for vp, v in s["viewports"].items()}}
        for r, s in summary["recs"].items()}, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser(description="uxcheck — визуальная и синхрон-проверка плеера")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--rec", nargs="*", type=int, default=[5, 6, 7])
    ap.add_argument("--asr", default=None, help="конкретный прогон (forced|google|whisper)")
    ap.add_argument("--viewports", nargs="*", default=DEFAULT_ORDER,
                    choices=list(VIEWPORTS.keys()))
    ap.add_argument("--samples", type=int, default=12, help="точек для сверки синхрона")
    ap.add_argument("--shots-only", action="store_true", help="только скрины (без сверки по выборке)")
    ap.add_argument("--out", default=None, help="папка для скринов (по умолчанию scratchpad)")
    a = ap.parse_args()

    out = Path(a.out) if a.out else Path(
        "/tmp/claude-1000/-home-abd-development-synchronized/"
        "79c4ba81-b69b-45b4-8494-4a66ea98a119/scratchpad/uxshots")
    run(a.base, a.rec, a.asr, a.viewports, a.samples, out, a.shots_only)


if __name__ == "__main__":
    sys.exit(main())
