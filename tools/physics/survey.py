"""列出源码与模板的调用方，并在隔离回放中统计每个入口的调用次数。"""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from gametest import function, records
from prepare import ROOT
from replay import Case, execute, suite

CALL = re.compile(r"(?:^|\s)\$?function\s+([^\s{]+)")
MEASURE = {
    "object/_iter_motion",
    "object/_iter_motion_slow_mov",
    "object/_iter_motion_slow_key",
    "point/_iter_motion",
    "object/_regular",
    "vehicle/_regular",
    "test_car/_regular",
    "object/_apply_friction",
    "tensor/_calc_inertia",
    "box_object/_apply_couple",
    "object/_set_angular",
    "cublock/_iter_cpoints",
    "vehicle/_iter_cpoints_c",
    "test_car/_iter_cpoints_c",
    "block/_iter_cpoints_c",
    "_detect_slope",
    "_detect_material",
    "_detect_box",
    "water/response",
    "test_boat/engine/main",
    "vehicle/engine/main",
    "object/_sync_motion",
    "test_car/_get",
    "test_car/_store",
    "cubox/_sync_motion",
}


def family(name, text):
    bare = name.split(":")[-1]
    leaf = bare.split("/")[-1].removesuffix(".mcfi")
    if (
        "/test/" in name
        or bare.startswith(("test/", "test_coord/"))
        or leaf.startswith("_test")
        or leaf.endswith("_test")
    ):
        return "测试与样例调度"
    if leaf == "main_loop" and "simulator" in name:
        return "模块调度或专用运算"
    if leaf == "main_spectator":
        return "载具动力与控制"
    if leaf in {"main_scale", "_calc_scale"}:
        return "显示与同步"
    if leaf in {"_poke_here_i", "main_l", "_flatten_velocity"}:
        return "力学与介质响应"
    if leaf == "_calc_chunk_range":
        return "碰撞点几何与世界查询"
    if leaf in {
        "_anchor_to",
        "_get",
        "_store",
        "_model",
        "_proj",
        "_print",
        "init",
        "_init",
        "_zero",
        "_new",
        "_del",
        "_class",
        "_consts",
        "set",
        "set_operation",
    }:
        return "构造与状态搬运"
    if "_regular" in leaf or "regular_cross" in name or "/regular/" in name:
        return "接触姿态修正"
    if any(
        k in name
        for k in (
            "iter_motion",
            "angular",
            "_rotate",
            "_apply_spin",
            "_set_rotation",
            "_set_roll",
            "spin/",
            "rand_quat",
        )
    ):
        return "运动学与姿态"
    if "tensor" in name or "inertia" in name:
        return "惯性张量"
    if any(
        k in name
        for k in (
            "cpoint",
            "detect_point",
            "_detect",
            "check_material",
            "solid/",
            "slope_block/",
            "block_reader/",
        )
    ):
        return "碰撞点几何与世界查询"
    if (
        "response" in leaf
        or "receive" in leaf
        or any(
            k in name
            for k in (
                "impulse",
                "couple",
                "friction",
                "grab_layer",
                "bounce_layer",
                "liquid/",
                "water/",
                "lava/",
                "shift",
            )
        )
    ):
        return "力学与介质响应"
    if "engine/" in name or "control/" in name or "damp" in name:
        return "载具动力与控制"
    if "sync" in name or "display" in name or "render" in name or "shader/" in name or "sound/" in name:
        return "显示与同步"
    if "function math:quat" in text or "function math:uvw" in text:
        return "运动学与姿态"
    if any(op in text for op in (" *= ", " /= ", " %= ")):
        return "专用数值组合"
    return "模块调度或专用运算"


EVIDENCE = {
    "运动学与姿态": "相位、向量长度和整段姿态的直接公式对照",
    "碰撞点几何与世界查询": "点积及六输出碰撞点试验。实体选择、方块读取、面分类由游戏执行",
    "接触姿态修正": "实测姿态模型、留出回放及独立的整刻准入结果。多面骰按各自的离散面选择，连续修正对应姿态和投影对照",
    "惯性张量": "可变六分量张量的二次型对照",
    "力学与介质响应": "摩擦、浮力、点积和六输出接触响应对照，保留响应标志和实体写入",
    "载具动力与控制": "运动变化部分使用阻尼和投影对照，玩家输入和启停阈值保留",
    "专用数值组合": "按所列命令与下游函数复核，未取得比直接运算更优的模型证据",
    "构造与状态搬运": "记录、读写或构造实例，物理运算按调用目标单独评估",
    "显示与同步": "实体、插值和显示写入仍须执行，坐标运算对应投影对照",
    "测试与样例调度": "运行场景或验证调用目标，不替换测试断言和调度",
    "模块调度或专用运算": "按调用目标评估计算，保留生命周期和分派",
}
PROFILE_SCENES = ("fall", "flat", "slope", "step", "dice", "impact", "water", "slow")


def scan():
    files = {}
    for f in sorted((ROOT / "data").glob("*/function/**/*.mcfunction")):
        rel = f.relative_to(ROOT)
        name = rel.parts[1] + ":" + Path(*rel.parts[3:]).with_suffix("").as_posix()
        files[name] = f
    for f in sorted((ROOT / "memory_storage").glob("*/templates/**/*.mcfi")):
        files["MOT/" + f.relative_to(ROOT / "memory_storage").as_posix()] = f
    static, callers = {}, defaultdict(set)
    for name, f in files.items():
        lines = [
            line.strip()
            for line in f.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        body = "\n".join(lines)
        calls = sorted(set(CALL.findall(body.replace("\\\n", " "))))
        for callee in calls:
            callers[callee].add(name)
            if name.startswith("MOT/") and callee.startswith("$(module_prefix)"):
                base = name.split("/templates/")[0]
                callers[base + "/templates/" + callee.removeprefix("$(module_prefix)") + ".mcfi"].add(name)
        signature = hashlib.sha256(body.encode()).hexdigest()[:16]
        static[name] = (family(name, body), signature, calls)
    return files, static, callers


def profile(work):
    files, static, callers = scan()
    cases = [Case(scene, 19, 2 if scene == "impact" else 1, 200) for scene in PROFILE_SCENES]
    entries = suite(cases, False)
    overrides, reset, report, setup = {}, [], [], []
    runtime = [n for n in files if not n.startswith("MOT/")]
    for idx, name in enumerate(runtime):
        path = files[name]
        rel = path.relative_to(ROOT).as_posix()
        body = path.read_text(encoding="utf-8")
        count = f"#p{idx} lab"
        first = f"scoreboard players add {count} 1\n"
        reset.append(f"scoreboard players reset {count}")
        fields = f"{{id:{idx},ms:0}}"
        if name.split(":")[1] in MEASURE:
            timer = f"lab:p{idx}"
            if any(line.startswith("$") for line in body.splitlines()):
                raise ValueError(f"计时包装不能改变宏输入：{name}")
            setup.append(f"stopwatch create {timer}")
            reset.append(f"scoreboard players set #t{idx} lab 0")
            function(entries, f"lab:profile/p{idx}", [body])
            overrides[rel] = (
                first + f"stopwatch restart {timer}\n"
                f"execute store success score #s{idx} lab store result score #r{idx} lab run function lab:profile/p{idx}\n"
                f"execute store result score #dt{idx} lab run stopwatch query {timer} 1000\n"
                f"scoreboard players operation #t{idx} lab += #dt{idx} lab\n"
                f"execute if score #s{idx} lab matches 0 run return fail\nreturn run scoreboard players get #r{idx} lab\n"
            ).encode()
        else:
            overrides[rel] = (first + body).encode()
        commands = [
            f"data modify storage lab:io p set value {fields}",
            f"execute store result storage lab:io p.n int 1 run scoreboard players get {count}",
            "execute store result storage lab:io p.case int 1 run scoreboard players get #case lab",
        ]
        if name.split(":")[1] in MEASURE:
            commands.append(
                f"execute store result storage lab:io p.ms int 1 run scoreboard players get #t{idx} lab"
            )
        commands.append("function lab:profile/log with storage lab:io p")
        function(entries, f"lab:profile/r{idx}", commands)
        report.append(f"execute if score {count} matches 1.. run function lab:profile/r{idx}")
    function(entries, "lab:profile/log", ["$say VVE_COUNT [$(case),$(id),$(n),$(ms)]"])
    function(entries, "lab:profile/reset", reset)
    function(entries, "lab:profile/report", report)
    entries["data/lab/function/init.mcfunction"] += ("\n" + "\n".join(setup) + "\n").encode()
    for i in range(len(cases)):
        start = f"data/lab/function/case_{i}/start.mcfunction"
        end = f"data/lab/function/case_{i}/end.mcfunction"
        entries[start] += b"function lab:profile/reset\n"
        entries[end] = b"function lab:profile/report\n" + entries[end]
    dest, lines = execute(work, "profile", cases, False, entries, overrides)
    counts = defaultdict(dict)
    for ci, idx, n, ms in records(lines, "COUNT"):
        counts[runtime[idx]][cases[ci].scene] = {"calls": n, "inclusive_ms": ms}
    return write_inventory(dest, files, static, callers, counts)


def write_inventory(dest, files, static, callers, counts):
    out = dest / "inventory.tsv"
    with out.open("w", encoding="utf-8", newline="") as stream:
        w = csv.writer(stream, delimiter="\t")
        w.writerow(
            [
                "入口",
                "文件",
                "职责",
                "函数正文 SHA-256 前缀",
                "静态调用方（含模板）",
                "调用目标（含动态宏）",
                "各场景调用及含子调用毫秒",
                "替换评估依据",
            ]
        )
        for n, f in files.items():
            group, sig, calls = static[n]
            w.writerow(
                [
                    n,
                    f.relative_to(ROOT).as_posix(),
                    group,
                    sig,
                    ",".join(sorted(callers[n])),
                    ",".join(calls),
                    json.dumps(counts.get(n, {}), separators=(",", ":")),
                    EVIDENCE[group],
                ]
            )
    print(out, len(files), sum(bool(v) for v in counts.values()), flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    profile(p.parse_args().directory.resolve())
