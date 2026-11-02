"""按物体和物理步对齐两个游戏回放，输出与原算法的偏差。"""
import argparse
import csv
import json
import math
import itertools

import numpy as np
from pathlib import Path


def read_frames(path):
    with path.open(encoding="utf-8") as stream:
        return {(r["case"], r["tick"], r["id"]): r for r in
                ({k: int(v) for k, v in row.items()} for row in csv.DictReader(stream))}


def distance(a, b, keys):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in keys)) / 10000


def angle(a, b):
    x = [a["quat_"+c] for c in "xyzw"]
    y = [b["quat_"+c] for c in "xyzw"]
    norm = math.sqrt(sum(v*v for v in x) * sum(v*v for v in y))
    if norm == 0:
        raise ValueError("姿态为零四元数")
    return math.degrees(2 * math.acos(min(1, abs(sum(u*v for u,v in zip(x, y))) / norm)))


HALVES = {"dice": (.25,.25,.25), "roll": (.25,.25,.25), "rock": (.5,.3,1), "impact": (.25,.25,.25), "flat": (.5,.3,1),
          "slope": (.6,.4,1), "slow": (.6,.4,1), "step": (.5,.3,1), "water": (.5,.3,1)}


def box(row, scene):
    center = np.array([row[a] for a in "xyz"]) / 10000
    q = np.array([row["quat_"+a] for a in "xyzw"], dtype=float)
    x,y,z,w = q / np.linalg.norm(q)
    axes = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    half = np.array(HALVES.get(scene,(0,0,0)))
    return center, axes, half


def penetration(row, scene, peer=None):
    center, axes, half = box(row, scene)
    corners = center + (np.array(list(itertools.product((-1,1), repeat=3))) * half) @ axes.T
    depth = 0.
    cell_x, cell_z = 8 + row["id"] % 8 * 8, 8 + row["id"] // 8 * 8
    for x,y,z in corners:
        if not (0 <= x < 76 and 0 <= z < 76):
            continue
        height = 81.
        if scene == "step" and cell_x+1 <= x < cell_x+4 and cell_z-2 <= z < cell_z+3:
            height = 82.
        if scene == "slope" and cell_x-1 <= x < cell_x+2 and cell_z-2 <= z < cell_z+3:
            height = 82 - (x % 1)
        depth = max(depth, height-y)
    if peer is not None:
        other, other_axes, other_half = box(peer, scene)
        normals = [*axes.T, *other_axes.T, *(np.cross(a,b) for a in axes.T for b in other_axes.T)]
        overlaps = []
        for normal in normals:
            norm = np.linalg.norm(normal)
            if norm > 1e-7:
                n = normal/norm
                overlaps.append(half @ abs(n @ axes) + other_half @ abs(n @ other_axes) - abs(n @ (other-center)))
        depth = max(depth, min(overlaps))
    return float(depth)

def compare(original, trial):
    base, sample = read_frames(original / "frames.csv"), read_frames(trial / "frames.csv")
    cases = json.loads((original / "timings.json").read_text(encoding="utf-8"))
    other = json.loads((trial / "timings.json").read_text(encoding="utf-8"))
    if [{k:v for k,v in c.items() if k != "us"} for c in cases] != [{k:v for k,v in c.items() if k != "us"} for c in other]:
        raise ValueError("场景参数不同")
    if base.keys() != sample.keys():
        raise ValueError("缺少对应的物体或物理步")
    stats = {}
    for key, a in base.items():
        i, tick, _ = key
        b = sample[key]
        group = stats.setdefault(str(i), dict(scene=cases[i]["scene"], position=0., velocity=0., angle_deg=0.,
                                                at_200=0., angle_200=0., rest_drift=0., extra_depth=0., flags=0))
        pos = distance(a, b, "xyz")
        rot = angle(a, b)
        group["position"] = max(group["position"], pos)
        group["velocity"] = max(group["velocity"], distance(a, b, ("vx", "vy", "vz")))
        group["angle_deg"] = max(group["angle_deg"], rot)
        group["flags"] += int((a["stable"], a["surface"]) != (b["stable"], b["surface"]))
        if tick == 200:
            group["at_200"] = max(group["at_200"], pos)
            group["angle_200"] = max(group["angle_200"], rot)
        if tick == 1200:
            group["rest_drift"] = max(group["rest_drift"], pos)
        peer_key = (i, tick, key[2] ^ 1)
        paired = cases[i]["scene"] == "impact"
        db = penetration(b, cases[i]["scene"], sample.get(peer_key) if paired else None)
        da = penetration(a, cases[i]["scene"], base.get(peer_key) if paired else None)
        group["extra_depth"] = max(group["extra_depth"], db - da)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.trial), indent=2))
