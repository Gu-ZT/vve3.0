"""从原版姿态求解器采集车辆和普通刚体的输入输出。"""
import argparse
import json
from pathlib import Path

import numpy as np

from gametest import function, records
from prepare import ROOT
from replay import Case, execute, suite

POSE_FILES = {
    "data/vve/function/vehicle/_regular.mcfunction": "vehicle",
    "data/vve_examples/function/test_car/_regular.mcfunction": "vehicle",
    "data/vve/function/object/_regular.mcfunction": "object",
}
AXES = [axis+"vec_"+c for axis in "ijk" for c in "xyz"]
NORMAL = ["nvec_"+c for c in "xyz"]
QUAT = ["quat_"+c for c in "xyzw"]
INPUTS = [*NORMAL, *AXES, *QUAT]


def original_pose(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    begin = source.index("function math:iquat/_nvec_to")
    stop = source.rindex("function math:quat/_touvw") + len("function math:quat/_touvw")
    return source[:begin], source[begin:stop], source[stop:]


def record(entries):
    for side, fields in (("in", INPUTS), ("out", QUAT)):
        commands = [f"execute store result storage lab:io sample.{side}_{f} int 1 run scoreboard players get {f} int" for f in fields]
        function(entries, "lab:read_"+side, commands)
    fields = ["group","tick","family","u","v","w", *["in_"+f for f in INPUTS], *["out_"+f for f in QUAT]]
    function(entries, "lab:sample_log", ["$say VVE_SAMPLE ["+",".join("$("+f+")" for f in fields)+"]"])
    for family, code in (("vehicle",1),("object",2)):
        commands = [f"data modify storage lab:io sample set value {{family:{code},u:0,v:0,w:0}}",
                    "execute store result storage lab:io sample.group int 1 run scoreboard players get #case lab",
                    "execute store result storage lab:io sample.tick int 1 run scoreboard players get #tick lab"]
        if family == "vehicle":
            commands += [f"execute store result storage lab:io sample.{c} int 1 run scoreboard players get stemp_{c} int" for c in "uvw"]
        function(entries, "lab:before_"+family, commands+["function lab:read_in"])
    function(entries, "lab:after", ["function lab:read_out", "function lab:sample_log with storage lab:io sample"])


def quat_product(a, b):
    v, w = a[..., :3], a[..., 3:]
    u, k = b[..., :3], b[..., 3:]
    return np.concatenate((v*k + u*w + np.cross(v,u), w*k - (v*u).sum(axis=-1,keepdims=True)),axis=-1)


def probes(entries, seeds, count):
    # 合成探针覆盖不同平面和朝向，同一随机种子的整组探针只属于一个分区。
    _, exact, _ = original_pose(next(iter(POSE_FILES)))
    function(entries, "lab:oracle/pose", ["function math:quat/_touvw", "function lab:before_object",
                                         exact, "function lab:after"])
    all_groups = []
    for group, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        setup = [f"scoreboard players set #case lab {1000+group}"]
        for tick in range(count):
            yaw = rng.uniform(-np.pi,np.pi)
            tilt = rng.uniform(-.70,.70)
            plane = rng.uniform(-np.pi,np.pi)
            half = tilt / 2
            axis = np.array([np.cos(plane),0,np.sin(plane)])
            q = quat_product(np.array([0,np.sin(yaw/2),0,np.cos(yaw/2)]), np.r_[axis*np.sin(half),np.cos(half)])
            spread = rng.uniform(-.40,.40)
            n = np.array([np.sin(spread)*np.cos(plane),np.cos(spread),np.sin(spread)*np.sin(plane)])
            setup.append(f"scoreboard players set #tick lab {tick}")
            for key, value in zip(QUAT+NORMAL, np.r_[q,n]):
                setup.append(f"scoreboard players set {key} int {round(value*10000)}")
            setup.append("function lab:oracle/pose")
        function(entries, f"lab:oracle/g{group}", setup)
        all_groups.append(f"execute as 0-0-0-0-0 run function lab:oracle/g{group}")
    function(entries,"lab:oracle/all",all_groups)


def collect(work):
    seeds = [11,23,37,47,61,73,89,101,113,127,139,151]
    scenes = ("flat","slope","dice","impact","step","water")
    cases = [Case(s, seed, 2 if s == "impact" else 1, 80) for seed in seeds for s in scenes]
    entries = suite(cases, False)
    record(entries)
    probes(entries, seeds, 96)
    function(entries,"lab:done",["function lab:oracle/all","say VVE_DONE"])
    overrides = {}
    for path,family in POSE_FILES.items():
        before, original, after = original_pose(path)
        overrides[path] = (before+f"function lab:before_{family}\n"+original+"\nfunction lab:after"+after).encode()
    out, lines = execute(work, "samples", cases, False, entries, overrides)
    values = np.asarray(records(lines,"SAMPLE"),dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 26 or not set(values[:,2]) == {1,2}:
        raise ValueError(f"姿态记录缺失：{values.shape}")
    groups = {i: [c.scene,c.seed] for i,c in enumerate(cases)}
    groups.update({1000+i:["oracle",s] for i,s in enumerate(seeds)})
    np.savez_compressed(out/"samples.npz", rows=values, groups=json.dumps(groups))
    print(out/"samples.npz", values.shape, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    collect(p.parse_args().directory.resolve())
