"""准备锁定版本的 26.3 服务器与数据包。"""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = {
    "math3.1": ("187aa724c8df9f16393b9f8b1c8bbe712378e36c", "f720feec99e8bca2405db8cc15a0afe4920bdffc877a67e3f14966c8bc55abca"),
    "math3.1_lalib": ("81e5d7a0083269d2d591e985d2e36154631006f9", "d7dcd8b399834c4194151c51dd39a75a6e8579820d84e862f4c328d8c80fe4eb"),
    "math3.1_gelib": ("cf7b385c0f8348da438f00d0deb886fea8181d62", "f9e8d12f8bfcc76b4a644a00e351818169e50137db40c9bc434462bb32147a0a"),
}
SERVER_SHA1 = "33680f5f2ac32864d6d7cf5e56a705fdb3e05f4c"
SUFFIXES = {".json", ".mcfunction", ".nbt"}


def download(url, path, algorithm, digest):
    def checksum(file):
        with file.open("rb") as stream:
            return hashlib.file_digest(stream, algorithm).hexdigest()

    if path.exists():
        if checksum(path) != digest:
            raise ValueError(f"缓存校验失败，请移走后重试：{path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
    try:
        print(f"下载 {path.name}", flush=True)
        with urllib.request.urlopen(url, timeout=90) as response, pending.open("wb") as target:
            shutil.copyfileobj(response, target)
        if checksum(pending) != digest:
            raise ValueError(f"下载校验失败：{url}")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def save_pack(path, entries):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
    try:
        with zipfile.ZipFile(pending, "w") as dest:
            for name, content in entries:
                dest.writestr(zipfile.ZipInfo(name), content, compress_type=zipfile.ZIP_DEFLATED)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def resources(archive, prefix):
    for name in sorted(archive.namelist()):
        if not name.startswith(prefix):
            continue
        rel = name[len(prefix):]
        if rel == "pack.mcmeta":
            meta = json.loads(archive.read(name))
            yield rel, json.dumps({"pack": {"description": meta["pack"]["description"],
                                            "min_format": [121, 0], "max_format": [121, 0]}}).encode()
        elif rel == "pack.png" or (rel.startswith("data/") and Path(rel).suffix in SUFFIXES):
            yield rel, archive.read(name)


def vve_resources():
    for name in ("pack.mcmeta", "pack.png"):
        yield name, (ROOT / name).read_bytes()
    for path in sorted((ROOT / "data").rglob("*")):
        if path.is_file() and path.suffix in SUFFIXES:
            yield path.relative_to(ROOT).as_posix(), path.read_bytes()


def prepare(directory, server=False):
    cache = directory / "cache"
    packs = directory / "packs"
    for folder in (cache, packs):
        folder.mkdir(parents=True, exist_ok=True)
    if server:
        download(f"https://piston-data.mojang.com/v1/objects/{SERVER_SHA1}/server.jar",
                 directory / "server.jar", "sha1", SERVER_SHA1)
    for name, (revision, digest) in DEPENDENCIES.items():
        archive = cache / f"{name}-{revision}.zip"
        download(f"https://codeload.github.com/xiaodou8593/{name}/zip/{revision}", archive, "sha256", digest)
        with zipfile.ZipFile(archive) as upstream:
            save_pack(packs / f"{name}.zip", resources(upstream, f"{name}-{revision}/"))
    save_pack(packs / "vve.zip", vve_resources())
    return packs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="隔离的工作目录")
    parser.add_argument("--server", action="store_true", help="同时下载 Mojang 26.3 服务端")
    args = parser.parse_args()
    print(prepare(args.directory.resolve(), args.server))
