"""重新加载正式包后，检查数学前置、ID 谓词以及普通和拆分物理步。"""

import argparse
import json
from pathlib import Path

from gametest import function, run


def smoke(work):
    entries = {"data/minecraft/tags/function/load.json": json.dumps({"values": ["lab:loaded"]}).encode()}
    function(
        entries,
        "lab:loaded",
        [
            "execute if data storage lab:io awaiting_reload run schedule function lab:verify 5t",
        ],
    )
    function(entries, "lab:begin", ["forceload add -16 -16 31 31", "schedule function lab:reload 20t"])
    function(entries, "lab:reload", ["data modify storage lab:io awaiting_reload set value 1b", "reload"])
    function(
        entries,
        "lab:verify",
        [
            "data remove storage lab:io awaiting_reload",
            "function math:_init",
            "function math:_init_la",
            "function math:_init_ge",
            "function vve:_init",
            "execute unless score math_init_version int matches 315 run return run say VVE_FAIL math",
            "execute unless score math_la_init_version int matches 315 run return run say VVE_FAIL lalib",
            "execute unless score math_ge_init_version int matches 315 run return run say VVE_FAIL gelib",
            "execute unless score vve_init_version int matches 304 run return run say VVE_FAIL vve",
            "function vve_examples:test_car/init",
            "function vve_examples:green_car/init",
            "execute positioned 8.0 90 8.0 rotated 0 0 run function vve_examples:test_car/_summon_here",
            "execute as @e[tag=vve_test_car,limit=1] run function vve_examples:test_car/main_c",
            "execute unless score @e[tag=vve_test_car,limit=1] x matches 80000 run return run say VVE_FAIL center",
            "execute unless score @e[tag=vve_test_car,limit=1] vy matches ..-1 run return run say VVE_FAIL gravity",
            "scoreboard players set @e[tag=vve_test_car,limit=1] vve_id 7",
            "scoreboard players set tempid int 7",
            "execute as @e[tag=vve_test_car,limit=1] unless predicate vve:match_id run return run say VVE_FAIL id_equal",
            "scoreboard players set tempid int 8",
            "execute as @e[tag=vve_test_car,limit=1] if predicate vve:match_id run return run say VVE_FAIL id_unequal",
            "execute positioned 10.0 90 10.0 rotated 0 0 run function vve_examples:green_car/_summon_here",
            "scoreboard players set inv_dt int 4",
            "scoreboard players set @e[tag=result,limit=1] vx 400",
            *(["execute as @e[tag=result,limit=1] run function vve_examples:green_car/main_slow_mov"] * 3),
            "execute as @e[tag=result,limit=1] run function vve_examples:green_car/main_slow_key",
            "execute unless score @e[tag=result,limit=1] x matches 100400 run return run say VVE_FAIL slow",
            "say VVE_DONE",
        ],
    )
    output, messages = run(work, "smoke", entries, 1000)
    if any("] VVE_FAIL " in line for line in messages):
        raise RuntimeError(f"入口验证失败：{output}")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    smoke(parser.parse_args().directory.resolve())
