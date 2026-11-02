"""把训练权重和同输入的直接公式分别导出为独立试验覆盖包。"""
import argparse
import json
from pathlib import Path

from gametest import function, resource
from prepare import ROOT, save_pack
from samples import POSE_FILES, original_pose


def score(name):
    return {"type":"score","target":{"type":"fixed","name":name},"score":"int"}


def flt(name):
    return {"type":"from_int","input":score(name)}


def op(kind,*args):
    if kind in {"add","mul","max","min","length"}:
        return {"type":kind,"inputs":list(args)}
    if kind in {"div","sub","floor_div"}:
        return {"type":kind,"left":args[0],"right":args[1]}
    return {"type":kind,"input":args[0]}


def affine(nodes, weights, bias):
    terms = [n if w == 1 else op("mul",w,n) for n,w in zip(nodes,weights) if w != 0]
    if bias or not terms:
        terms.append(bias)
    return terms[0] if len(terms)==1 else op("add",*terms)


def network(model, node):
    nodes=[node]
    for k,(matrix,bias) in enumerate(zip(model["w"],model["b"])):
        nodes=[affine(nodes,weights,b) for weights,b in zip(zip(*matrix),bias)]
        if k+1<len(model["w"]):
            nodes=[op("max",0,n) for n in nodes]
    return nodes[0]


def build(models, mode, families=("vehicle","object"), flat_vehicle=False):
    entries, replacements = {}, {}

    def assign(name, value, dest, kind="integer", scale=1):
        resource(entries, "lab:"+name, "context_"+("int" if kind=="integer" else "float")+"_provider", value)
        suffix = f" {scale}" if scale!=1 else ""
        return f"execute store result score {dest} int run compute default {kind} lab:{name}{suffix}"

    local = [assign("pose/local_"+axis,
                    op("floor_div",op("add",*[op("mul",score(axis+"vec_"+c),score("nvec_"+c)) for c in "xyz"]),10000),
                    "lab_d"+axis) for axis in "ijk"]
    for family in families:
        # 该子程序的其余写回（位置、角速度、接触点列表）在原调用方继续执行。
        fallback = "return run function lab:pose/original"
        wrapper = ["scoreboard players set lab_cur int 1", "execute unless score lab_prev int matches 1 run "+fallback]
        if family=="vehicle":
            wrapper += ["execute unless score temp_surface int matches 1 run "+fallback,
                        "execute unless score lab_triangle int matches 1 run "+fallback]
        if family=="vehicle" and flat_vehicle:
            # 限定到近静止且三点共面的水平支撑，坡面变化和滑动仍走原求解。
            for key, bounds in (("nvec_x","-5..5"),("nvec_y","9995..10005"),("nvec_z","-5..5"),
                                ("vx","-300..300"),("vz","-300..300")):
                wrapper.append(f"execute unless score {key} int matches {bounds} run "+fallback)
        wrapper.extend(local)
        for axis in "ijk":
            for label,bounds in (("p","9200..10010"),("n","-10010..-9200")):
                wrapper.append(f"execute if score lab_d{axis} int matches {bounds} run return run function lab:pose/{family}_{axis}{label}")
        function(entries,"lab:pose/"+family,[*wrapper,"function lab:pose/original"])
        # 有符号主轴决定离散分支，网络只估算进入该分支后的连续修正。
        for j,axis in enumerate("ijk"):
            for label,sign in (("p",1),("n",-1)):
                a,b=(j+1)%3,(j+2)%3
                fields=["lab_rx","lab_ry","lab_rz"]
                lines=[f"execute if score lab_d{'ijk'[a]} int matches 0 if score lab_d{'ijk'[b]} int matches 0 run return run function math:quat/_touvw",
                       f"scoreboard players operation lab_align int = lab_d{axis} int"]
                if sign<0:
                    lines.append("scoreboard players operation lab_align int *= -1 int")
                gain = (op("div",1,op("sqrt",op("mul",2,op("add",1,op("mul",.0001,flt("lab_align"))))))
                        if mode=="formula" else network(models[family],op("max",0,op("mul",.00125,op("sub",10000,flt("lab_align"))))))
                lines.append(assign(f"pose/{family}_gain",gain,"lab_gain","float",1000000))
                lines.append(f"scoreboard players set {fields[j]} int 0")
                for target,src,sgn in ((a,b,-sign),(b,a,sign)):
                    lines.append(assign(f"pose/r_{axis}{label}_{target}",
                                        op("mul",sgn*.000001,flt("lab_gain"),flt("lab_d"+"ijk"[src])),
                                        fields[target],"float"))
                # 原轴表含定点误差，平面稳定时须由已量化的切向分量恢复单位四元数。
                sq=op("add",*[op("mul",flt(fields[k]),flt(fields[k])) for k in (a,b)])
                w=op("sqrt",op("max",0,op("sub",100000000,sq)))
                lines.append(assign(f"pose/w_{axis}",w,"lab_rw","float"))
                lines.append("function lab:pose/apply")
                function(entries,f"lab:pose/{family}_{axis}{label}",lines)
    matrix = [
        [(3,0,1),(0,3,1),(1,2,1),(2,1,-1)], [(3,1,1),(1,3,1),(2,0,1),(0,2,-1)],
        [(3,2,1),(2,3,1),(0,1,1),(1,0,-1)], [(3,3,1),(0,0,-1),(1,1,-1),(2,2,-1)],
    ]
    apply=[]
    for c,terms in zip("xyzw",matrix):
        products=[op("mul",score("quat_"+"xyzw"[u]),score("lab_r"+"xyzw"[v])) for u,v,s in terms]
        total=affine(products,[s for u,v,s in terms],0)
        apply.append(assign("pose/q"+c,op("floor_div",total,10000),"lab_q"+c))
    apply.extend(f"scoreboard players operation quat_{c} int = lab_q{c} int" for c in "xyzw")
    function(entries,"lab:pose/apply",[*apply,"function math:quat/_touvw"])
    for path,family in POSE_FILES.items():
        if family not in families:
            continue
        before, original, after=original_pose(path)
        if family=="vehicle":
            before=before.replace("function math:vec/_cross_fvec_high",
                "function math:vec/_cross_fvec_high\nscoreboard players set lab_triangle int 1\n"
                "execute if score vec_x int matches 0 if score vec_y int matches 0 if score vec_z int matches 0 run scoreboard players set lab_triangle int 0")
        function(entries,"lab:pose/original",[original])
        replacements[path]=(before+"function lab:pose/"+family+after).encode()
    # main 在根实体上运行，姿态求解在共享世界实体上运行。按实例记住上一物理步是否接触。
    for f in (ROOT/"data").glob("*/function/**/main*.mcfunction"):
        s=f.read_text(encoding="utf-8")
        if "/_regular" not in s or "/_get" not in s:
            continue
        relevant=(("vehicle" in families and any(k in s for k in ("vehicle/_regular","test_car/_regular"))) or
                  ("object" in families and any(k in s for k in ("object/_regular","cublock/_regular"))))
        if relevant:
            prefix="execute store success score lab_prev int if entity @s[tag=lab_recent]\nscoreboard players set lab_cur int 0\n"
            tail="\nexecute if score lab_cur int matches 1 run tag @s add lab_recent\nexecute if score lab_cur int matches 0 run tag @s remove lab_recent\n"
            replacements[f.relative_to(ROOT).as_posix()]=(prefix+s+tail).encode()
    return entries,replacements


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("models",type=Path)
    p.add_argument("directory",type=Path)
    p.add_argument("--families",nargs="+",choices=["vehicle","object"],default=["vehicle","object"])
    p.add_argument("--flat-vehicle",action="store_true",help="仅在近静止水平支撑时使用车辆模型")
    a=p.parse_args()
    m=json.loads(a.models.read_text(encoding="utf-8"))
    a.directory.mkdir(parents=True,exist_ok=True)
    for mode in ("formula","network"):
        extra,overrides=build(m,mode,a.families,a.flat_vehicle)
        save_pack(a.directory/(mode+".zip"),{**extra,**overrides}.items())
