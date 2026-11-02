"""在游戏内重放留出样本，校验推理、负数换算与原入口回退。"""
import argparse
import json
from pathlib import Path

import numpy as np

from export import build
from gametest import function, records, run
from samples import INPUTS
from train import HOLD


def verify(work, models):
    with np.load(work/"results/samples/samples.npz") as src:
        rows=src["rows"]
        groups=json.loads(str(src["groups"]))
    selected=[i for i,r in enumerate(rows) if groups[str(r[0])][1] in HOLD]
    entries={}
    for mode in ("formula","network"):
        assets,_=build(models,mode)
        for path,data in assets.items():
            entries[path.replace("data/lab/",f"data/check_{mode}/")] = data.replace(b"lab:",f"check_{mode}:".encode())
    function(entries,"lab:begin",["forceload add -1 -1 16 16","schedule function lab:init 20t"])
    function(entries,"lab:init",["function math:_init","function math:_init_la","function vve:_init",
                                 "schedule function lab:batch_0 5t"])
    function(entries,"lab:report",["$say VVE_ORACLE [$(id),$(mode),$(first),$(x),$(y),$(z),$(w),$(ax),$(ay),$(az),$(vx),$(vy),$(vz)]"])
    batches=[selected[i:i+20] for i in range(0,len(selected),20)]
    for batch,ids in enumerate(batches):
        calls=[]
        for i in ids:
            row=rows[i]
            family={1:"vehicle",2:"object"}[row[2]]
            constants=[f"scoreboard players set {k} int {v}" for k,v in zip(INPUTS,row[6:22])]
            lines=[]
            for m,mode in enumerate(("formula","network","original")):
                first=int(i%13==0)
                lines += constants+[f"scoreboard players set lab_prev int {1-first}",
                              "scoreboard players set temp_surface int 1","scoreboard players set lab_triangle int 1",
                              *[f"scoreboard players set v{c} int {speed}" for c,speed in zip("xyz",(-73,19,41))],
                              f"function check_{mode}:pose/{family}" if mode!="original" else "function check_formula:pose/original",
                              f"data modify storage lab:io result set value {{id:{i},mode:{m},first:{first}}}"]
                lines += [f"execute store result storage lab:io result.{c} int 1 run scoreboard players get quat_{c} int"
                          for c in "xyzw"]
                # 车辆位置修正使用首个接触点的局部坐标。相同质心和位移响应下，
                # 转回世界坐标的差就是这一步对齐接触点引入的质心差。刚体无该位移写回。
                lines += [f"scoreboard players set {c} int {int(value)}" for c,value in zip("uvw",row[3:6])]
                lines += [f"scoreboard players set {c} int 0" for c in "xyz"]+["function math:uvw/_tovec"]
                for c in "xyz":
                    lines += [f"execute store result storage lab:io result.a{c} int 1 run scoreboard players get vec_{c} int",
                              f"execute store result storage lab:io result.v{c} int 1 run scoreboard players get v{c} int"]
                lines.append("function lab:report with storage lab:io result")
            function(entries,f"lab:oracle/{i}",lines)
            calls.append(f"function lab:oracle/{i}")
        function(entries,f"lab:oracle/batch_{batch}",[*calls, "tp @s 0.0 0.0 0.0"])
        function(entries,f"lab:batch_{batch}",[f"execute as 0-0-0-0-0 run function lab:oracle/batch_{batch}",
                 f"schedule function lab:batch_{batch+1} 1t" if batch+1<len(batches) else "say VVE_DONE"])
    dest,messages=run(work,"oracle",entries,len(batches)+45)
    values=records(messages,"ORACLE")
    if len(values)!=len(selected)*3 or {(r[0],r[1]) for r in values}!={(i,m) for i in selected for m in (0,1,2)}:
        raise ValueError("同输入的推理结果不完整")
    teacher={r[0]:r for r in values if r[1]==2}
    results={}
    for mode,label in enumerate(("formula","network")):
        for family,code in (("vehicle",1),("object",2)):
            matched=[r for r in values if r[1]==mode and rows[r[0],2]==code]
            q=np.asarray([r[3:7] for r in matched],float)
            target=rows[[r[0] for r in matched],22:26].astype(float)
            angles=2*np.degrees(np.arccos(np.clip(abs((q*target).sum(axis=1))/(np.linalg.norm(q,axis=1)*np.linalg.norm(target,axis=1)),0,1)))
            initial=np.array([bool(r[2]) for r in matched])
            results[label+"/"+family]=dict(n=len(q),max_deg=float(max(angles)),p95_deg=float(np.quantile(angles,.95)),
                fallback_mismatches=int((q[initial]!=target[initial]).any(axis=1).sum()),
                position=float(max(np.linalg.norm(np.subtract(r[7:10],teacher[r[0]][7:10]))/10000 for r in matched)),
                velocity=float(max(np.linalg.norm(np.subtract(r[10:13],teacher[r[0]][10:13]))/10000 for r in matched)))
    (dest/"accuracy.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(results,flush=True)
    return all(r["max_deg"]<=.25 and r["position"]<=.005 and r["velocity"]<=.005 and
               r["fallback_mismatches"]==0 for r in results.values())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    a=p.parse_args()
    passed=verify(a.directory.resolve(),json.loads((a.directory/"results/models.json").read_text(encoding="utf-8")))
    raise SystemExit(0 if passed else 2)
