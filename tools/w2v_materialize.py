import sys, json, time
sys.path.insert(0,'/app/src'); sys.path.insert(0,'/app/service')
from pathlib import Path
from recitations.models import Recitation, AsrRun
from recitations import pipeline
from player import build_data
q=pipeline._quran()
for rid in [5,11]:
    p=Path(f"/app/work/w2v_{rid}.json")
    if not p.exists(): print(f"rec{rid}: нет {p}"); continue
    sync_map=json.loads(p.read_text())
    rec=Recitation.objects.get(id=rid)
    # записать sync-map в run_dir (как делает run_one)
    out=pipeline.run_dir(rid,"w2v"); out.mkdir(parents=True,exist_ok=True)
    (out/"sync-map.json").write_text(json.dumps(sync_map,ensure_ascii=False,indent=2))
    data=build_data(sync_map,q,rec.audio_filename)
    audio_dur=(rec.meta or {}).get("duration") or 0
    tl_end=data["timeline"][-1]["t"] if data.get("timeline") else 0
    data["duration"]=round(audio_dur or tl_end)
    if sync_map.get("char_timeline"): data["char_timeline"]=sync_map["char_timeline"]
    wt=data.get("word_timeline") or []; tl=data.get("timeline") or []
    meta=dict(sync_map.get("meta",{}))
    if "coverage" in meta: meta["aligned_ratio"]=meta.pop("coverage")
    dur_for_cov=audio_dur or tl_end
    sp,ratio=pipeline._speech_time_coverage(wt,dur_for_cov)
    bins=pipeline._audio_time_coverage(wt,dur_for_cov)
    run,_=AsrRun.objects.get_or_create(recitation=rec,recognizer="w2v")
    run.data=data
    run.metrics={**meta,"coverage":ratio,"speech_sec":sp,
                 "silence_sec":round(max(0.0,(dur_for_cov or 0)-sp),1),
                 "coverage_bins":bins,"wt":len(wt),"tl":len(tl),"duration":round(audio_dur or tl_end)}
    run.status=AsrRun.Status.READY; run.stage=""; run.error=""
    run.save()
    print(f"rec{rid}: w2v прогон READY, cov={ratio}, wt={len(wt)}, аятов={len(tl)}")
