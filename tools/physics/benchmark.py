"""交错比较同一批场景的原实现、解析公式和模型。"""
import argparse
import hashlib
import itertools
import json
import random
import zipfile
from pathlib import Path

import numpy as np

from replay import Case, SCENES, execute


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


def benchmark(work, name, formula, network, target, scenes, rounds):
    if rounds < 12 or target not in scenes:
        raise ValueError("须至少 12 轮，并包含事先选定的目标场景")
    cases = [Case(s, 19, n, 1000, 200) for s in scenes for n in (1,16,64)]
    files = {"original": None, "formula": formula, "network": network}
    candidates = {g: variant(p) for g,p in files.items()}
    output = work / "results" / name
    output.mkdir(parents=True, exist_ok=True)
    config = {"target": target, "cases": [vars(c) for c in cases], "rounds": rounds,
              "variants": {g: hashlib.sha256(p.read_bytes()).hexdigest() if p else None for g,p in files.items()}}
    (output / "protocol.json").write_text(json.dumps(config,indent=2), encoding="utf-8")
    orders = list(itertools.permutations(files))
    rng = random.Random(2603)
    rows = []
    for round_no in range(rounds):
        if round_no % 6 == 0:
            rng.shuffle(orders)
        row = {}
        for group in orders[round_no % 6]:
            print(f"{round_no+1}/{rounds} {group}", flush=True)
            dest, _ = execute(work, f"{name}/round-{round_no+1}/{group}", cases, False, *candidates[group])
            row[group] = json.loads((dest/"timings.json").read_text(encoding="utf-8"))
        rows.append(row)
    summary = intervals(rows, cases)
    for count in (1,16,64):
        if not any(s["count"] == count for s in summary):
            raise ValueError("缺少物体数量分组")
    # 目标场景以 64 个活动物体为主测组。其他组按无明显退化检验。
    for s in summary:
        target_load = s["scene"] == target and s["count"] == 64
        upper = max(s["ratio_original"][1], s["ratio_formula"][1])
        s["pass"] = upper < 1 if target_load else upper <= 1.02
    (output / "summary.json").write_text(json.dumps(summary,indent=2), encoding="utf-8")
    return all(s["pass"] for s in summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("name")
    p.add_argument("formula", type=Path)
    p.add_argument("network", type=Path)
    p.add_argument("--target", choices=list(SCENES), required=True)
    p.add_argument("--scenes", nargs="+", choices=list(SCENES), default=list(SCENES))
    p.add_argument("--rounds", type=int, default=12)
    a = p.parse_args()
    print("通过" if benchmark(a.directory.resolve(), a.name, a.formula, a.network, a.target, a.scenes, a.rounds) else "未通过")
