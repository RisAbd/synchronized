"""M5 — генератор плеера. sync-map + канон + аудио -> самодостаточный player.html.

Плеер: крупный оригинальный арабский текст (RTL), подсветка текущего аята и авто-скролл
в такт аудио по `timeline` из sync-map. Аудио подключается по относительному пути (кладём
player.html рядом с аудио). Тема (свет/тьма) подхватывается системная.

CLI:  python3 player.py <sync-map.json> <audio_src> [out.html]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from quran import Quran, map_editions


def _ayah_spans(timeline: list[dict]) -> dict[int, tuple[int, int]]:
    """Для каждой суры из timeline — диапазон затронутых аятов (min, max)."""
    spans: dict[int, tuple[int, int]] = {}
    for e in timeline:
        s, a = e["surah"], e["ayah"]
        if s not in spans:
            spans[s] = (a, a)
        else:
            lo, hi = spans[s]
            spans[s] = (min(lo, a), max(hi, a))
    return spans


def build_data(sync_map: dict, quran: Quran, audio_src: str) -> dict:
    """Данные для плеера: аудио + timeline + разделы текста + оглавление (chapters).
    Единый формат и для standalone-плеера, и для веб-фронта."""
    timeline = sync_map["timeline"]
    spans = _ayah_spans(timeline)

    # разделы текста: по каждой затронутой суре — аяты от min до max.
    # words — разбивка отображаемого текста на слова; индекс слова совпадает с word_index
    # канона (нормализация не меняет число/порядок слов), поэтому по нему подсвечиваем слово.
    sections = []
    for surah in sorted(spans):
        lo, hi = spans[surah]
        s = quran.surah(surah)
        ayat = []
        for a in range(lo, hi + 1):
            v = quran.verse(surah, a)
            words = v.text.split()
            item = {"ayah": a, "text": v.text, "words": words}
            # П9: вторая редакция (Diyanet) для переключения текста в плеере. word_timeline
            # индексирован по словам Tanzil (wi) → отдаём карту wi→[индексы слов Diyanet],
            # чтобы подсветка легла на слова Diyanet даже при ином дроблении. forced не трогаем.
            if getattr(v, "text_diyanet", ""):
                dwords = v.text_diyanet.split()
                item["text_diyanet"] = v.text_diyanet
                item["words_diyanet"] = dwords
                item["dmap"] = map_editions(words, dwords)
            ayat.append(item)
        sections.append({"surah": surah, "title": s.title, "ayat": ayat})

    # оглавление: точки смены суры (для навигации-«chapters»)
    chapters = []
    for e in timeline:
        if not chapters or chapters[-1]["surah"] != e["surah"]:
            chapters.append({"t": e["t"], "surah": e["surah"], "ayah": e["ayah"],
                             "title": quran.surah(e["surah"]).title})

    return {
        "audio": audio_src,
        "timeline": [{"t": e["t"], "surah": e["surah"], "ayah": e["ayah"]} for e in timeline],
        # t_end (если есть, forced его даёт) тащим на фронт: плеер по нему замораживает
        # karaoke-заливку слова на паузах (не «ползёт» прогресс, пока чтец молчит/договаривает).
        "word_timeline": [{k: w[k] for k in ("t", "t_end", "surah", "ayah", "wi") if k in w}
                          for w in sync_map.get("word_timeline", [])],
        "sections": sections,
        "chapters": chapters,
    }


def build(sync_map: dict, quran: Quran, audio_src: str) -> str:
    payload = json.dumps(build_data(sync_map, quran, audio_src), ensure_ascii=False)
    return _HTML.replace("/*__DATA__*/", payload)


_HTML = r"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>synchronized — плеер</title>
<style>
  :root { color-scheme: light dark; --bg:#faf8f2; --fg:#1c1a15; --muted:#9a9384;
          --hl-bg:#fff2c2; --hl-fg:#111; --accent:#c8a020; --card:#fff; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12110d; --fg:#e9e4d6; --muted:#6f6a5c; --hl-bg:#3a3413;
            --hl-fg:#fff6d0; --accent:#e0bd45; --card:#1b1a14; }
  }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg);
    font-family: "Amiri","Scheherazade New","Noto Naskh Arabic","Times New Roman",serif; }
  #bar { position:sticky; top:0; z-index:10; background:var(--card);
    border-bottom:1px solid rgba(128,128,128,.2); padding:10px 14px;
    display:flex; gap:12px; align-items:center; direction:ltr; }
  #bar audio { flex:1; height:38px; }
  #pos { font:600 14px system-ui,sans-serif; color:var(--muted); white-space:nowrap; }
  #text { max-width:900px; margin:0 auto; padding:24px 18px 60vh; }
  .surah-title { text-align:center; font-size:1.4rem; color:var(--accent);
    margin:34px 0 18px; padding-bottom:8px; border-bottom:1px solid rgba(128,128,128,.2); }
  .ayah { font-size:2.15rem; line-height:2.5; padding:2px 6px; border-radius:8px;
    transition: background .25s, color .25s; cursor:default; }
  .ayah .num { font-size:1.1rem; color:var(--muted); margin:0 6px;
    font-family:system-ui,sans-serif; }
  .ayah.active { background:var(--hl-bg); color:var(--hl-fg); }
  #hint { position:fixed; bottom:0; left:0; right:0; text-align:center;
    font:12px system-ui,sans-serif; color:var(--muted); padding:6px; direction:ltr;
    background:linear-gradient(transparent,var(--bg) 40%); }
</style>
</head>
<body>
  <div id="bar">
    <audio id="audio" controls preload="metadata"></audio>
    <span id="pos">—</span>
  </div>
  <div id="text"></div>
  <div id="hint">synchronized · авто-прокрутка по чтению</div>
<script>
const DATA = /*__DATA__*/;
const audio = document.getElementById('audio');
const posEl = document.getElementById('pos');
const textEl = document.getElementById('text');
audio.src = DATA.audio;

// рендер текста
const elById = {};
for (const sec of DATA.sections) {
  const h = document.createElement('div');
  h.className = 'surah-title';
  h.textContent = 'سورة ' + sec.title;
  textEl.appendChild(h);
  for (const v of sec.ayat) {
    const d = document.createElement('span');
    d.className = 'ayah';
    d.id = 'a-' + sec.surah + '-' + v.ayah;
    d.innerHTML = v.text + ' <span class="num">﴿' + toArabic(v.ayah) + '﴾</span> ';
    textEl.appendChild(d);
    elById[sec.surah + ':' + v.ayah] = d;
  }
}
function toArabic(n){ return String(n).replace(/[0-9]/g, d => '٠١٢٣٤٥٦٧٨٩'[d]); }

// timeline отсортирован по времени; ищем текущую запись бинарно
const TL = DATA.timeline;
let activeKey = null, activeEl = null, userScrolling = false, scrollTimer = null;

function currentIndex(t){
  let lo=0, hi=TL.length-1, res=-1;
  while(lo<=hi){ const m=(lo+hi)>>1; if(TL[m].t<=t){res=m; lo=m+1;} else hi=m-1; }
  return res;
}
function fmt(t){ t=Math.max(0,t|0); return (t/60|0)+':'+String(t%60).padStart(2,'0'); }

function update(){
  const t = audio.currentTime;
  const i = currentIndex(t);
  if (i < 0){ posEl.textContent = fmt(t); return; }
  const e = TL[i];
  const key = e.surah + ':' + e.ayah;
  posEl.textContent = sectitle(e.surah) + ' ' + e.surah + ':' + e.ayah + '  ·  ' + fmt(t);
  if (key === activeKey) return;
  if (activeEl) activeEl.classList.remove('active');
  activeKey = key;
  activeEl = elById[key];
  if (activeEl){
    activeEl.classList.add('active');
    if (!userScrolling)
      activeEl.scrollIntoView({behavior:'smooth', block:'center'});
  }
}
function sectitle(s){ const x=DATA.sections.find(z=>z.surah===s); return x?('سورة '+x.title):('сура '+s); }

audio.addEventListener('timeupdate', update);
audio.addEventListener('seeked', update);
// если пользователь сам скроллит — не дёргаем автоскроллом ~3с
addEventListener('wheel', pause, {passive:true});
addEventListener('touchmove', pause, {passive:true});
function pause(){ userScrolling=true; clearTimeout(scrollTimer);
  scrollTimer=setTimeout(()=>userScrolling=false, 3000); }
</script>
</body>
</html>"""


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python3 player.py <sync-map.json> <audio_src> [out.html]", file=sys.stderr)
        sys.exit(1)
    sync_map = json.loads(Path(sys.argv[1]).read_text())
    audio_src = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "work/player.html"

    q = Quran.load()
    html = build(sync_map, q, audio_src)
    Path(out).write_text(html)
    print(f"player -> {out}  (аудио: {audio_src})")
