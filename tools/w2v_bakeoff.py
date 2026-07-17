import sys, json, glob, traceback
sys.path.insert(0,"/home/abd/development/synchronized/src")
sys.path.insert(0,"/home/abd/development/synchronized/service")
import w2v_align, torch
V=json.load(open("/home/abd/development/synchronized/work/verses.json"))
CAP=0.6
def speech_cov(wt,dur):
    pts=sorted((w for w in wt if w.get("t") is not None),key=lambda w:w["t"]); ivs=[]
    for i,w in enumerate(pts):
        t0=float(w["t"]); te=w.get("t_end")
        if te is not None and float(te)>t0: t1=float(te)
        else:
            nxt=pts[i+1]["t"] if i+1<len(pts) else t0+CAP
            t1=t0+min(CAP,max(0.0,nxt-t0)) if nxt>t0 else t0+CAP
        t0=max(0,min(t0,dur)); t1=max(t0,min(t1,dur)); ivs.append((t0,t1))
    ivs.sort(); c=0.0; cs=ce=None
    for a,b in ivs:
        if cs is None: cs,ce=a,b
        elif a<=ce: ce=max(ce,b)
        else: c+=ce-cs; cs,ce=a,b
    if cs is not None: c+=ce-cs
    return c/dur if dur else 0
MODELS=["", "elgeish/wav2vec2-large-xlsr-53-arabic",
        "TBOGamer22/wav2vec2-quran-phonetics",
        "IbrahimSalah/Wav2vecLarge_quran_syllables_recognition"]
for mdl in MODELS:
    name=mdl or "jonatasgrosman(default)"
    w2v_align._ALIGN_MODEL=mdl; w2v_align._align_model=None; w2v_align._align_meta=None
    torch.cuda.empty_cache()
    row=[]
    for rid in [5,11]:
        d=V[str(rid)]; verses=[tuple(x) for x in d["verses"]]; windows=d["windows"]; dur=d["dur"]
        ap=sorted(glob.glob(f"/home/abd/development/synchronized/media/rec/{rid}/audio.*"))[0]
        try:
            sm=w2v_align.align(ap,verses,windows=windows)
            wt=sm["word_timeline"]; m=sm["meta"]
            deg=sum(1 for w in wt if w.get("t_end") is not None and abs(w["t_end"]-w["t"])<0.03)
            cov=speech_cov(wt,dur)
            rahman=""
            if rid==5 and len(wt)>1: rahman=f"الرحمن={(wt[1].get('t_end') or wt[1]['t'])-wt[1]['t']:.2f}с"
            row.append(f"rec{rid} cov={cov:.3f} выровн={m['aligned_units']}/{m['ref_words']} вырожд={deg} {rahman}")
        except Exception as e:
            row.append(f"rec{rid} ОШИБКА: {type(e).__name__}: {str(e)[:80]}")
    print(f"### {name}"); [print("   ",r) for r in row]
print("BAKEOFF_DONE")
