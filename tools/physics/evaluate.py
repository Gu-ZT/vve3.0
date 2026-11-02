"""比较留出场景的完整原版回放，记录逐项淘汰原因。"""
import argparse
import json
from pathlib import Path

from benchmark import variant
from compare import compare
from replay import execute, fixtures, suite
from gametest import records


def evaluate(work, name, folder, seeds):
    result={}
    for seed in seeds:
        cases=fixtures(seed)
        path=work/"results"/name/str(seed)
        for label in ("original","formula","network"):
            extra,overrides = ({},{}) if label=="original" else variant(folder/(label+".zip"))
            count_pack = suite(cases,True)
            for family in ("vehicle","object"):
                for axis in "ijk":
                    for sign in "pn":
                        key=f"data/lab/function/pose/{family}_{axis}{sign}.mcfunction"
                        if key in extra:
                            first, tail=extra[key].split(b"\n",1)
                            extra[key]=first+f"\nscoreboard players add #hits_{family} lab 1\n".encode()+tail
            for i in range(len(cases)):
                start=f"data/lab/function/case_{i}/start.mcfunction"
                end=f"data/lab/function/case_{i}/end.mcfunction"
                count_pack[start]+=b"scoreboard players set #hits_vehicle lab 0\nscoreboard players set #hits_object lab 0\n"
                count_pack[end]+=f"execute store result storage lab:io hits.v int 1 run scoreboard players get #hits_vehicle lab\nexecute store result storage lab:io hits.o int 1 run scoreboard players get #hits_object lab\ndata modify storage lab:io hits.i set value {i}\nfunction lab:hits with storage lab:io hits\n".encode()
            count_pack["data/lab/function/hits.mcfunction"]=b"$say VVE_HITS [$(i),$(v),$(o)]\n"
            dest,lines=execute(work, f"{name}/{seed}/{label}", cases, True,{**count_pack,**extra}, overrides)
            (dest/"hits.json").write_text(json.dumps(records(lines,"HITS")),encoding="utf-8")
            if label=="original":
                continue
            stats=compare(path/"original", path/label)
            for stat in stats.values():
                reasons=[]
                if stat["scene"] in ("flat","slope","step","slow","rock") and (stat["at_200"]>.02 or stat["angle_200"]>1):
                    reasons.append("200 刻轨迹超差")
                if stat["rest_drift"]>.01:
                    reasons.append("静置漂移超差")
                if stat["extra_depth"]>.005:
                    reasons.append("穿透深度超差")
                if stat["flags"]:
                    reasons.append("接触或静止状态不同")
                stat["reasons"]=reasons
            result[f"{seed}/{label}"]=stats
        (path.parent/"accuracy.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(seed,{k:[s["scene"] for s in result[f"{seed}/{k}"].values() if s["reasons"]] for k in ("formula","network")},flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("name")
    p.add_argument("variants",type=Path)
    p.add_argument("--seeds",nargs="+",type=int,default=[19,173,197])
    a=p.parse_args()
    result=evaluate(a.directory.resolve(),a.name,a.variants,a.seeds)
    passed=all(not s["reasons"] for scenes in result.values() for s in scenes.values())
    raise SystemExit(0 if passed else 2)
