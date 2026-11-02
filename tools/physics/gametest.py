"""在临时世界运行 26.3 原版 GameTest，并核对探针确实执行。"""
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from prepare import DEPENDENCIES, prepare, save_pack


def function(entries, name, commands):
    namespace, path = name.split(":")
    entries[f"data/{namespace}/function/{path}.mcfunction"] = ("\n".join(commands) + "\n").encode()


def resource(entries, name, kind, value):
    namespace, path = name.split(":")
    entries[f"data/{namespace}/{kind}/{path}.json"] = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def run(work, name, entries, ticks, overrides=None):
    """每次从同一组依赖建立新世界。只有声明过的 VVE 文件可被试验版本覆盖。"""
    packs = prepare(work, server=True)
    output = work / "results" / name
    output.mkdir(parents=True, exist_ok=True)
    entries["pack.mcmeta"] = json.dumps({"pack": {"description": "VVE physics test",
                                                   "min_format": [121, 0], "max_format": [121, 0]}}).encode()
    resource(entries, "lab:physics", "test_instance", {
        "type": "minecraft:function", "function": "minecraft:always_pass", "environment": "lab:physics",
        "structure": "minecraft:empty", "setup_ticks": ticks + 100, "max_ticks": 5,
    })
    resource(entries, "lab:physics", "test_environment", {"type": "minecraft:function", "setup": "lab:begin"})
    with tempfile.TemporaryDirectory(prefix="vve-world-", dir=work) as temp:
        folder = Path(temp)
        selected = folder / "packs"
        selected.mkdir()
        for archive in [f"{key}.zip" for key in DEPENDENCIES] + ["vve.zip"]:
            if archive == "vve.zip" and overrides:
                with zipfile.ZipFile(packs / archive) as original:
                    missing = set(overrides) - set(original.namelist())
                    if missing:
                        raise ValueError(f"覆盖了未知入口：{missing}")
                    save_pack(selected / archive, ((n, overrides[n] if n in overrides else original.read(n))
                                                   for n in original.namelist()))
            else:
                try:
                    os.link(packs / archive, selected / archive)
                except OSError:
                    shutil.copyfile(packs / archive, selected / archive)
        save_pack(selected / "lab.zip", entries.items())
        report = output / "gametest.xml"
        report.unlink(missing_ok=True)
        args = ["java", "-Xms1G", "-Xmx3G", "-XX:ActiveProcessorCount=4",
                "-DbundlerMainClass=net.minecraft.gametest.Main", "-jar", "server.jar",
                "--packs", str(selected), "--tests", "lab:physics", "--report", str(report),
                "--universe", str(folder / "world")]
        log = output / "server.log"
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(args, cwd=work, stdout=stream, stderr=subprocess.STDOUT, check=True)
        messages = log.read_text(encoding="utf-8", errors="replace").splitlines()
        errors = [line for line in messages if "/ERROR]" in line or "Failed to load function" in line]
        if errors or not report.is_file() or ET.parse(report).find(".//failure") is not None:
            raise RuntimeError(f"游戏内运行失败：{log}\n" + "\n".join(errors[:5]))
        if not any("] VVE_DONE" in line for line in messages):
            raise RuntimeError(f"测试未执行到末尾：{log}")
        return output, messages


def records(messages, label):
    marker = "] VVE_" + label + " "
    return [json.loads(line.split(marker, 1)[1]) for line in messages if marker in line]
