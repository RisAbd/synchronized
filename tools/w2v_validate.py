import sys, json, glob
sys.path.insert(0,"/home/abd/development/synchronized/src")
sys.path.insert(0,"/home/abd/development/synchronized/service")
import w2v_align, torch
V=json.load(open("/home/abd/development/synchronized/work/verses.json"))
CAP=0.6
def speech_cov(wt, dur):
    pts=sorted((w for w in wt if w.get("t") is not None), key=lambda w:w["t"]); ivs=[]
    for i,w in enumerate(pts):
        t0=float(w["t"]); te=w.get("t_end")
        if te is not None and float(te)>t0: t1=float(te)
        else:
            nxt=pts[i+1]["t"] if i+1<len(pts) else t0+CAP
            t1=t0+min(CAP,max(0.0,nxt-t0)) if nxt>t0 else t0+CAP
        t0=max(0,min(t0,dur)); t1=max(t0,min(t1,dur)); ivs.append((t0,t1))
    ivs.sort(); cov=0.0; cs=ce=None
    for a,b in ivs:
        if cs is None: cs,ce=a,b
        elif a<=ce: ce=max(ce,b)
        else: cov+=ce-cs; cs,ce=a,b
    if cs is not None: cov+=ce-cs
    return cov/dur if dur else 0
FCOV={5:0.493,7:0.428,9:0.509,11:0.457,14:0.442,10:0.504}
for rid in [5,7,11,14,10,9]:
    d=V[str(rid)]; verses=[tuple(x) for x in d["verses"]]; windows=d["windows"]; dur=d["dur"]
    ap=sorted(glob.glob(f"/home/abd/development/synchronized/media/rec/{rid}/audio.*"))[0]
    try:
        sm=w2v_align.align(ap, verses, windows=windows)
    except Exception as e:
        import traceback; print(f"rec{rid}: ОШИБКА {type(e).__name__}: {e}"); traceback.print_exc(); torch.cuda.empty_cache(); continue
    m=sm["meta"]; cov=speech_cov(sm["word_timeline"], dur)
    print(f"rec{rid} dur={dur:.0f}s: w2v cov={cov:.3f} (forced={FCOV[rid]}) "
          f"| выровн {m['aligned_units']}/{m['ref_words']} interp={m['interpolated']} snap={m['snapped_to_silence']}")
    json.dump(sm, open(f"/home/abd/development/synchronized/work/w2v_{rid}.json","w"), ensure_ascii=False)
print("DONE")
