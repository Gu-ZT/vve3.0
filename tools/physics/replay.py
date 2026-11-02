"""在真实游戏刻上回放固定种子的物理场景，或测量整个游戏刻。"""
import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from gametest import function, records, run

SCENES = {
    "fall": "vve:point", "flat": "vve_examples:test_car", "slope": "vve_examples:green_car",
    "step": "vve_examples:test_car", "dice": "vve_examples:dice_6", "impact": "vve:cublock",
    "water": "vve_examples:test_boat", "slow": "vve_examples:green_car",
}
FIELDS = "tick id x y z vx vy vz quat_x quat_y quat_z quat_w surface stable".split()
RESULT = "@e[tag=result,limit=1]"
MARKER = "0-0-0-0-0"


@dataclass(frozen=True)
class Case:
    scene: str
    seed: int
    count: int
    ticks: int
    warmup: int = 0


def spawn(case, index, rng):
    scene = case.scene
    module = SCENES[scene]
    ns, model = module.split(":")
    row = index // 2 if scene == "impact" else index
    x, z = 8.25 + row % 8 * 8, 8.25 + row // 8 * 8
    if scene == "impact":
        x += (index % 2) * 1.8
    y = {"fall": 90, "flat": 81.45, "slope": 82, "step": 81.4, "dice": 84,
         "impact": 82.1, "water": 84, "slow": 83}[scene] + rng.uniform(0, 0.07)
    yaw = rng.uniform(-18, 18)
    pitch = rng.uniform(-8, 8) if scene in ("dice", "slope", "flat", "water") else 0
    pos = f"{x:.5f} {y:.5f} {z:.5f}"
    commands = []
    if scene == "slope":
        commands.append(f"fill {int(x)-1} 81 {int(z)-2} {int(x)+1} 81 {int(z)+2} birch_trapdoor[facing=east]")
    if scene == "step":
        commands.append(f"fill {int(x)+1} 81 {int(z)-2} {int(x)+3} 81 {int(z)+2} stone")
    if scene == "fall":
        commands.append(f"data modify storage vve:io input set value {{center:[{x}d,{y:.5f}d,{z}d],velocity:[0d,0d,0d]}}")
    else:
        commands += [f"data modify storage {ns}:io input set from storage {ns}:class {model}_plate",
                     f"function {module}/_proj",
                     f"execute positioned {pos} rotated {yaw:.4f} {pitch:.4f} as {MARKER} run function vve:object/_anchor_to"]
        if scene in ("dice", "impact", "step"):
            speed = {"dice": 490, "impact": 900 if index % 2 == 0 else -900, "step": 1300}[scene]
            commands.append(f"scoreboard players set vx int {speed}")
        if scene == "dice":
            for axis in "xyz":
                commands.append(f"scoreboard players set angular_{axis} int {rng.randint(-110000, 110000)}")
            commands.append(f"execute as {MARKER} run function vve:object/_set_angular")
        commands += [f"function {module}/_model", f"data modify storage {ns}:io input set from storage {ns}:io result"]
    return commands + [f"execute positioned {pos} run function {module}/_new",
                       f"tag {RESULT} add lab_body", f"scoreboard players set {RESULT} lab_id {index}",
                       f"execute as {RESULT} on passengers run tag @s add lab_part",
                       f"execute as {RESULT} on passengers on passengers run tag @s add lab_part"]


def motion(scene):
    module = SCENES[scene]
    each = "execute as @e[tag=lab_body]"
    if scene == "slow":
        return ["scoreboard players set inv_dt int 4",
                "execute if score #tick lab matches 101.. run scoreboard players set inv_dt int 1",
                "scoreboard players set #phase lab 0", "scoreboard players operation #phase lab = #tick lab",
                "scoreboard players operation #phase lab %= inv_dt int",
                f"{each} unless score #phase lab matches 0 run function {module}/main_slow_mov",
                f"{each} if score #phase lab matches 0 run function {module}/main_slow_key",
                f"{each} if score #phase lab matches 0 run function {module}/main_force",
                f"{each} run function {module}/main_sync"]
    main = "main" if scene in ("impact", "fall") else "main_c"
    return [f"{each} run function {module}/{main}"]


def suite(cases, trace):
    entries = {}
    function(entries, "lab:begin", ["forceload add -16 -16 95 95", "schedule function lab:init 30t"])
    function(entries, "lab:init", ["function math:_init", "function math:_init_la", "function math:_init_ge",
             "function vve:_init", "scoreboard objectives add lab dummy", "scoreboard objectives add lab_id dummy",
             *[f"function {module}/init" for module in sorted(set(SCENES.values())) if module.startswith("vve_examples:")],
             "stopwatch create lab:wall", "schedule function lab:case_0/start 3t"])
    function(entries, "lab:done", ["say VVE_DONE"])
    frame = ["data modify storage lab:io row set value {surface:0,stable:0,quat_x:0,quat_y:0,quat_z:0,quat_w:10000}"]
    for field in FIELDS:
        if field in ("surface", "stable"):
            tag = "vve_surface" if field == "surface" else "stable"
            frame.append(f"execute if entity @s[tag={tag}] run data modify storage lab:io row.{field} set value 1")
        else:
            score = {"id": "@s lab_id", "tick": "#tick lab"}.get(field, f"@s {field}")
            frame.append(f"execute if score {score} matches -2147483648..2147483647 store result storage lab:io row.{field} int 1 run scoreboard players get {score}")
    frame += ["execute store result storage lab:io row.case int 1 run scoreboard players get #case lab",
              "function lab:frame_log with storage lab:io row"]
    function(entries, "lab:frame", frame)
    function(entries, "lab:frame_log", ["$say VVE_FRAME [$(case)," + ",".join("$("+key+")" for key in FIELDS) + "]"])
    function(entries, "lab:time_log", ["$say VVE_TIME [$(case),$(us),$(alive)]"])
    for i, case in enumerate(cases):
        prefix = f"lab:case_{i}/"
        end_tick = case.ticks + case.warmup
        rng = random.Random(case.seed)
        clears = ["kill @e[tag=lab_part]", "kill @e[tag=lab_body]",
                  "fill 0 81 0 75 84 75 air", "fill 0 80 0 75 80 75 stone"]
        if case.scene == "water":
            clears.append("fill 0 81 0 75 82 75 water")
        start = clears + ["function vve:_consts", f"scoreboard players set #case lab {i}",
                         "scoreboard players set #tick lab 0", "scoreboard players set input_w int 0",
                         "scoreboard players set input_s int 0"]
        for j in range(case.count):
            start.extend(spawn(case, j, rng))
        start += ["stopwatch restart lab:wall", f"schedule function {prefix}tick 1t"]
        function(entries, prefix + "start", start)
        step = ["scoreboard players add #tick lab 1", *motion(case.scene)]
        if trace:
            step.append("execute as @e[tag=lab_body] run function lab:frame")
        if case.warmup:
            step.append(f"execute if score #tick lab matches {case.warmup} run stopwatch restart lab:wall")
        step += [f"execute if score #tick lab matches {end_tick} run return run function {prefix}end",
                 f"schedule function {prefix}tick 1t"]
        function(entries, prefix + "tick", step)
        nxt = f"lab:case_{i+1}/start" if i+1 < len(cases) else "lab:done"
        function(entries, prefix + "end", [
            "execute store result storage lab:io timing.us int 1 run stopwatch query lab:wall 1000000",
            "execute store result storage lab:io timing.alive int 1 if entity @e[tag=lab_body]",
            f"data modify storage lab:io timing.case set value {i}",
            "function lab:time_log with storage lab:io timing", f"schedule function {nxt} 1t"])
    return entries


def execute(work, name, cases, trace, extra=None, overrides=None):
    entries = suite(cases, trace)
    if extra:
        entries.update(extra)
    duration = sum(c.ticks + c.warmup + 2 for c in cases)
    output, lines = run(work, name, entries, duration, overrides)
    times = records(lines, "TIME")
    if len(times) != len(cases) or any(i != k or us <= 0 or n != cases[k].count for k, (i, us, n) in enumerate(times)):
        raise ValueError(f"计时或存活实例数不符合场景设置：{times}")
    timings = [dict(**vars(c), us=us) for c, (_, us, _) in zip(cases, times)]
    (output / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    if trace:
        frames = records(lines, "FRAME")
        expected = {(i, t, j) for i, c in enumerate(cases) for t in range(1, c.ticks+c.warmup+1) for j in range(c.count)}
        if len(frames) != len(expected) or {tuple(row[:3]) for row in frames} != expected:
            raise ValueError(f"轨迹不完整：{output}")
        with (output / "frames.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(["case", *FIELDS])
            writer.writerows(sorted(frames))
    return output, lines


def fixtures(seed):
    return [Case(scene, seed, 2 if scene == "impact" else 1, 1200 if scene == "flat" else 200) for scene in SCENES]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()
    print(execute(args.directory.resolve(), args.name, fixtures(args.seed), True)[0])
