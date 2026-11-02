"""交错比较同一批场景的原实现、解析公式和模型。"""
import argparse
import hashlib
import itertools
import json
import random
import re
import zipfile
from pathlib import Path

import numpy as np

from prepare import ROOT, prepare
from replay import Case, SCENES, execute, suite
from gametest import records


def variant(path):
    if path is None:
        return {}, {}
    with zipfile.ZipFile(path) as z:
        extra, overrides = {}, {}
        for name in z.namelist():
            if name.startswith(("data/vve/", "data/vve_examples/")):
                overrides[name] = z.read(name)
            elif name.startswith("data/lab/"):
                extra[name] = z.read(name)
            else:
                raise ValueError(f"未知试验资源：{name}")
    return extra, overrides


def intervals(rows, cases):
    rng = np.random.default_rng(2603)
    result = []
    for i, case in enumerate(cases):
        item = dict(scene=case.scene, count=case.count)
        for group in ("original", "formula", "network"):
            item[group] = float(np.mean([row[group][i]["us"] / (1000*case.ticks) for row in rows]))
        # 共同重采样整轮，保留三个实现共享的机器状态与种子。
        ix = rng.integers(len(rows), size=(20000, len(rows)))
        for group in ("original", "formula"):
            trial = np.array([r["network"][i]["us"] for r in rows])
            ref = np.array([r[group][i]["us"] for r in rows])
            ratios = trial[ix].mean(axis=1) / ref[ix].mean(axis=1)
            item["ratio_"+group] = [float(x) for x in np.quantile(ratios, [.025,.975])]
        result.append(item)
    return result


CALL=re.compile(rb"(\bfunction\s+)((?:vve|vve_examples):[a-z0-9_./-]+)")


def isolate(label, path):
    extra, overrides=variant(path)
    namespace=("lab_"+label).encode()
    assets={k.replace("data/lab/","data/"+namespace.decode()+"/"):v.replace(b"lab:",namespace+b":")
            for k,v in extra.items()}
    sources={}
    changed=set()
    for ns in ("vve","vve_examples"):
        folder=ROOT/"data"/ns/"function"
        for f in folder.rglob("*.mcfunction"):
            key=(ns+":"+f.relative_to(folder).with_suffix("").as_posix()).encode()
            rel=f.relative_to(ROOT).as_posix()
            sources[key]=overrides.get(rel,f.read_bytes())
            if rel in overrides:
                changed.add(key)
    edges={key:{m[1] for m in CALL.findall(body)} for key,body in sources.items()}
    old_size=-1
    while len(changed)!=old_size:
        old_size=len(changed)
        changed.update(key for key,refs in edges.items() if refs & changed)
    names={key:namespace+b":"+key.replace(b":",b"/") for key in changed}

    def rewrite(data):
        return CALL.sub(lambda m:m[1]+names.get(m[2],m[2]),data.replace(b"lab:",namespace+b":"))

    for key,new in names.items():
        prefix,sub=new.decode().split(":")
        assets[f"data/{prefix}/function/{sub}.mcfunction"]=rewrite(sources[key])
    return assets,names


def trial(work,name,formula,network,target,scenes,rounds=12,check=False):
    if not check and rounds<12:
        raise ValueError("正式性能测试要求至少十二轮")
    if target not in scenes:
        raise ValueError("目标场景不在测试集")
    variants={"original":None,"formula":formula,"network":network}
    groups=[(s,n) for s in scenes for n in ((1,) if check else (1,16,64))]
    orders=list(itertools.permutations(variants))
    schedule=[]
    rng=random.Random(2603)
    for r in range(rounds):
        if r%6==0:
            rng.shuffle(orders)
        indexes=list(range(len(groups)))
        rng.shuffle(indexes)
        for j in indexes:
            for mode in orders[(r+j)%6]:
                schedule.append((r,j,mode))
    base=[Case(s,19,n,200 if check else 1000,0 if check else 200) for s,n in groups]
    cases=[base[j] for r,j,mode in schedule]
    entries=suite(cases,check)
    for mode,file in variants.items():
        if file is None:
            continue
        assets,names=isolate(mode,file)
        entries.update(assets)
        for i,(_,_,g) in enumerate(schedule):
            if g!=mode:
                continue
            path=f"data/lab/function/case_{i}/tick.mcfunction"
            entries[path]=CALL.sub(lambda m:m[1]+names.get(m[2],m[2]),entries[path])
    out=work/"results"/name
    out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").unlink(missing_ok=True)
    original_pack=prepare(work,server=True)/"vve.zip"
    protocol={"target":target,"groups":groups,"rounds":rounds,"schedule":schedule,
              "archives":{k:hashlib.sha256((p or original_pack).read_bytes()).hexdigest() for k,p in variants.items()},
              "warmup":base[0].warmup,"ticks":base[0].ticks}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    execute(work,name,cases,check,entries)
    return out, (True if check else summarize(out))


def summarize(out):
    """从完整服务端日志重建统计。缺少整轮、少了物体或序号重复都禁止判定通过。"""
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    rounds=protocol["rounds"]
    if rounds<12 or protocol["warmup"]!=200 or protocol["ticks"]!=1000:
        raise ValueError("计时协议不满足准入条件")
    groups=protocol["groups"]
    target=protocol["target"]
    if [target,64] not in groups:
        raise ValueError("缺少目标场景的 64 物体组")
    schedule=protocol["schedule"]
    expected={(r,j,m) for r in range(rounds) for j in range(len(groups)) for m in ("original","formula","network")}
    if len(schedule)!=len(expected) or {tuple(row) for row in schedule}!=expected:
        raise ValueError("三组运行次序不完整")
    logs=(out/"server.log").read_text(encoding="utf-8",errors="replace").splitlines()
    times=records(logs,"TIME")
    if len(times)!=len(schedule) or not any("] VVE_DONE" in line for line in logs):
        raise ValueError("服务器尚未完成所有测量")
    rows=[{m:[None]*len(groups) for m in ("original","formula","network")} for _ in range(rounds)]
    for seq,((r,j,m),(i,us,alive)) in enumerate(zip(schedule,times,strict=True)):
        if i!=seq or alive!=groups[j][1] or us<=0:
            raise ValueError(f"第 {seq} 个测量块无效")
        rows[r][m][j]={"us":us}
    base=[Case(s,19,n,protocol["ticks"],protocol["warmup"]) for s,n in groups]
    summary=intervals(rows,base)
    for row in summary:
        upper=max(row["ratio_original"][1],row["ratio_formula"][1])
        row["pass"]=upper<1 if (row["scene"],row["count"])==(target,64) else upper<=1.02
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return all(row["pass"] for row in summary)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("name")
    p.add_argument("formula",type=Path)
    p.add_argument("network",type=Path)
    p.add_argument("--target",choices=list(SCENES),required=True)
    p.add_argument("--scenes",nargs="+",choices=list(SCENES),required=True)
    p.add_argument("--rounds",type=int,default=12)
    p.add_argument("--check",action="store_true",help="仅验证同 JVM 内的版本切换与轨迹")
    a=p.parse_args()
    location,passed=trial(a.directory.resolve(),a.name,a.formula,a.network,a.target,a.scenes,
                          1 if a.check else a.rounds,a.check)
    print(location, "通过" if passed else "未通过")
    raise SystemExit(0 if passed else 2)
