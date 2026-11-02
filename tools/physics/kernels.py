"""对重复使用的连续数值核进行拟合，并在原版游戏内测量标量求值成本。"""

import argparse
import json
from pathlib import Path

import numpy as np

from export import affine, flt, op
from gametest import function, records, resource, run
from train import ARCHITECTURES

DIMENSIONS = {
    "phase": 1,
    "axis_length": 3,
    "projection": 6,
    "inertia": 9,
    "friction": 2,
    "buoyancy": 4,
    "damping": 4,
    "rigid_pose": 9,
    "contact_point": 18,
    "contact_response": 10,
}
FUSED = ("rigid_pose", "contact_point", "contact_response")
NAMES = [n for n in DIMENSIONS if n not in FUSED]


def quaternion(x, c, s, add, mul):
    return [add(mul(c, a), mul(s, b)) for a, b in zip(x[1:5], x[5:9])]


def basis(q, add, mul, sub):
    x, y, z, w = q
    xx, yy, zz, ww = [mul(a, a) for a in q]
    return [
        add(ww, xx, mul(-1, yy), mul(-1, zz)),
        mul(2, sub(mul(x, y), mul(z, w))),
        mul(2, add(mul(x, z), mul(y, w))),
        mul(2, add(mul(x, y), mul(z, w))),
        add(ww, yy, mul(-1, xx), mul(-1, zz)),
        mul(2, sub(mul(y, z), mul(x, w))),
        mul(2, sub(mul(x, z), mul(y, w))),
        mul(2, add(mul(y, z), mul(x, w))),
        add(ww, zz, mul(-1, xx), mul(-1, yy)),
    ]


def offset(x, add, mul):
    return [add(*(mul(a, b) for a, b in zip(x[j * 3 : j * 3 + 3], x[9:12]))) for j in range(3)]


def point_velocity(x, r, add, mul, sub):
    return [
        add(
            x[15 + j], sub(mul(x[12 + (j + 1) % 3], r[(j + 2) % 3]), mul(x[12 + (j + 2) % 3], r[(j + 1) % 3]))
        )
        for j in range(3)
    ]


def response(x, normal_speed, add, mul, sub, div):
    normal = x[:3]
    velocity = x[3:6]
    depth, mass, bounce_inv, friction = x[6:10]
    outgoing = div(mul(-1, normal_speed), bounce_inv)
    tangent_scale = sub(friction, 1)
    impulses = [
        mul(mass, add(mul(n, outgoing), mul(sub(v, mul(n, normal_speed)), tangent_scale)))
        for n, v in zip(normal, velocity)
    ]
    return [*[mul(n, depth) for n in normal], *impulses]


def operands(name, x, ops):
    add, mul, sub, div, sqrt, sin, cos, minimum = ops
    if name == "contact_response":
        speed = add(*(mul(a, b) for a, b in zip(x[:3], x[3:6])))
        return response(x, speed, add, mul, sub, div)
    if name == "rigid_pose":
        q = quaternion(x, cos(x[0]), sin(x[0]), add, mul)
        return [*q, *basis(q, add, mul, sub)]
    if name == "contact_point":
        r = offset(x, add, mul)
        return [*r, *point_velocity(x, r, add, mul, sub)]
    if name == "phase":
        return [sin(x[0]), cos(x[0])]
    if name == "axis_length":
        return [sqrt(add(*[mul(a, a) for a in x]))]
    if name == "projection":
        return [add(*[mul(x[i], x[i + 3]) for i in range(3)])]
    if name == "inertia":
        u, v, w, xx, yy, zz, xy, xz, yz = x
        return [
            add(
                mul(u, u, xx),
                mul(v, v, yy),
                mul(w, w, zz),
                mul(-2, u, v, xy),
                mul(-2, u, w, xz),
                mul(-2, v, w, yz),
            )
        ]
    if name == "friction":
        return [mul(*x)]
    if name == "buoyancy":
        area, depth, cap, coefficient = x
        return [mul(area, minimum(depth, cap), coefficient)]
    # 已确定阻尼力方向后，输入为速度、弹性倍率分子和分母、有符号阻尼力。
    return [add(div(mul(x[0], x[1]), x[2]), x[3])]


def inputs(name, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, DIMENSIONS[name]))
    if name == "contact_response":
        x[:, :3] /= np.linalg.norm(x[:, :3], axis=1, keepdims=True)
        x[:, 3:6] *= np.where((x[:, :3] * x[:, 3:6]).sum(axis=1) > 0, -1, 1)[:, None]
        x[:, 6:8] = abs(x[:, 6:8])
        x[:, 8:] = rng.uniform(0.4, 1, (n, 2))
    elif name == "rigid_pose":
        x[:, 0] *= np.pi
        a = x[:, 1:5]
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = x[:, 5:]
        b -= a * (a * b).sum(axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
    elif name == "contact_point":
        q = rng.normal(size=(n, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        x[:, :9] = np.column_stack(
            basis(q.T, lambda *a: sum(a), lambda *a: np.prod(np.broadcast_arrays(*a), axis=0), np.subtract)
        )
    elif name == "phase":
        x *= np.pi
    elif name == "inertia":
        x[:, :3] /= np.linalg.norm(x[:, :3], axis=1, keepdims=True)
        x[:, 3:6] = rng.uniform(0.2, 1, (n, 3))
        x[:, 6:] *= 0.05
    elif name == "friction":
        x[:, 1] = rng.uniform(0.3, 1, n)
    elif name == "buoyancy":
        x = abs(x)
    elif name == "damping":
        x[:, 1:3] = rng.uniform(0.4, 1, (n, 2))
    return np.round(x * 10000).astype(np.int32) / 10000


def labels(name, x):
    ops = (
        lambda *a: sum(a),
        lambda *a: np.prod(np.broadcast_arrays(*a), axis=0),
        np.subtract,
        np.divide,
        np.sqrt,
        np.sin,
        np.cos,
        np.minimum,
    )
    return np.column_stack(operands(name, x.T, ops))


def fit(x, y, shape):
    rng = np.random.default_rng(2603)
    widths = [x.shape[1], *shape, y.shape[1]]
    w = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(widths, widths[1:])]
    b = [rng.normal(0, 0.15, width) for width in widths[1:]]
    params = [*w, *b]
    m = [np.zeros_like(v) for v in params]
    v = [np.zeros_like(t) for t in params]
    for step in range(1, 1201):
        batch = rng.integers(len(x), size=96)
        act = [x[batch]]
        for j in range(len(w)):
            z = act[-1] @ w[j] + b[j]
            act.append(z if j == len(w) - 1 else np.maximum(0, z))
        g = (act[-1] - y[batch]) / len(batch)
        grads = [None] * len(params)
        for j in reversed(range(len(w))):
            grads[j] = act[j].T @ g
            grads[len(w) + j] = g.sum(axis=0)
            if j:
                g = (g @ w[j].T) * (act[j] > 0)
        for j, (par, grad) in enumerate(zip(params, grads)):
            m[j] = 0.9 * m[j] + 0.1 * grad
            v[j] = 0.999 * v[j] + 0.001 * grad**2
            par -= 0.006 * (m[j] / (1 - 0.9**step)) / (np.sqrt(v[j] / (1 - 0.999**step)) + 1e-8)
    return {"w": [a.tolist() for a in w], "b": [a.tolist() for a in b], "shape": list(shape)}


def infer(model, x):
    for j, (w, b) in enumerate(zip(model["w"], model["b"])):
        x = x @ np.asarray(w) + b
        if j + 1 < len(model["w"]):
            x = np.maximum(0, x)
    return x


def graph(entries, name, model, mode):
    nodes = [op("mul", 0.0001, flt(f"lab_x{i}")) for i in range(DIMENSIONS[name])]
    if mode == "formula":
        ops = tuple(
            (lambda *args, k=k: op(k, *args))
            for k in ("add", "mul", "sub", "div", "sqrt", "sin", "cos", "min")
        )
        outputs = operands(name, nodes, ops)
        if name == "axis_length":
            outputs = [op("length", *nodes)]
        stages = [outputs]

        def stored(stage, count):
            return [{"type": "storage", "storage": "lab:io", "path": f"h{stage}_{k}"} for k in range(count)]

        if name == "rigid_pose":
            c, s = stored(0, 2)
            q = stored(1, 4)
            stages = [
                [op("cos", nodes[0]), op("sin", nodes[0])],
                quaternion(nodes, c, s, ops[0], ops[1]),
                [*q, *basis(q, ops[0], ops[1], ops[2])],
            ]
        if name == "contact_response":
            speed = ops[0](*(ops[1](a, b) for a, b in zip(nodes[:3], nodes[3:6])))
            stages = [[speed], response(nodes, stored(0, 1)[0], ops[0], ops[1], ops[2], ops[3])]
        if name == "contact_point":
            r = stored(0, 3)
            stages = [offset(nodes, ops[0], ops[1]), [*r, *point_velocity(nodes, r, ops[0], ops[1], ops[2])]]
    else:
        stages = []
        for j, (w, b) in enumerate(zip(model["w"], model["b"])):
            outputs = [affine(nodes, weights, bias) for weights, bias in zip(zip(*w), b)]
            if j + 1 < len(model["w"]):
                outputs = [op("max", 0, n) for n in outputs]
            if mode == "cached" and j + 1 < len(model["w"]):
                stages.append(outputs)
                nodes = [
                    {"type": "storage", "storage": "lab:io", "path": f"h{j}_{k}"} for k in range(len(outputs))
                ]
            else:
                nodes = outputs
        stages.append(outputs)
    code = []
    for j, stage in enumerate(stages):
        for k, node in enumerate(stage):
            key = f"lab:k/{name}_{mode}_{j}_{k}"
            resource(entries, key, "context_float_provider", node)
            if j + 1 < len(stages):
                code.append(f"data modify storage lab:io h{j}_{k} set compute default float {key}")
            else:
                code.append(f"execute store result score lab_y{k} int run compute default float {key} 10000")
    function(entries, f"lab:k/{name}_{mode}", code)


def screen(work, fused=False):
    names = list(FUSED) if fused else NAMES
    experiment = "fused" if fused else "kernels"
    entries = {}
    output = work / "results" / experiment
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    expected_outputs = {}
    plan = []
    for i, name in enumerate(names):
        print(f"训练数值核 {i + 1}/{len(names)} {name}", flush=True)
        x = inputs(name, 4096, 11)
        valid = inputs(name, 1024, 23)
        hold = inputs(name, 2048, 47)
        y = labels(name, x)
        models = [fit(x, y, shape) for shape in ARCHITECTURES]
        best = min(models, key=lambda m: np.mean((infer(m, valid) - labels(name, valid)) ** 2))
        err = abs(infer(best, hold) - labels(name, hold))
        summary[name] = {
            "shape": best["shape"],
            "holdout_max": float(err.max()),
            "holdout_rms": float(np.sqrt((err**2).mean())),
            "max_game_units": 0,
            "validation": [
                {
                    "shape": m["shape"],
                    "max": float(abs(infer(m, valid) - labels(name, valid)).max()),
                    "rms": float(np.sqrt(np.mean((infer(m, valid) - labels(name, valid)) ** 2))),
                }
                for m in models
            ],
        }
        expected_outputs[name] = {
            "formula": labels(name, hold),
            "inline": infer(best, hold),
            "cached": infer(best, hold),
            "small": infer(models[0], hold),
        }
        # 同时给最小 4 单元模型计时，避免只拿昂贵的大模型同原公式比较。
        for mode, model in (("formula", None), ("inline", best), ("cached", best), ("small", models[0])):
            graph(entries, name, model, mode)
        for r in range(12):
            modes = ["formula", "inline", "cached", "small"]
            np.random.default_rng(2603 + 13 * i + r).shuffle(modes)
            for mode in modes:
                plan.append((name, r, mode, hold[r]))
    function(
        entries,
        "lab:begin",
        [
            "gamerule max_command_sequence_length 2147483647",
            "scoreboard objectives add int dummy",
            "stopwatch create lab:kernels",
            "schedule function lab:trial/0 5t",
        ],
    )
    for name in names:
        output_width = expected_outputs[name]["formula"].shape[1]
        function(
            entries,
            f"lab:log/{name}",
            ["$say VVE_KERNEL [$(id),$(us)," + ",".join(f"$(y{k})" for k in range(output_width)) + "]"],
        )
    for i, (name, r, mode, x) in enumerate(plan):
        call = f"function lab:k/{name}_{mode}"
        if r == 0:
            function(entries, f"lab:k/loop_{name}_{mode}", [call] * 32)
        # 每次真实读取动态记分板输入，长循环摊薄毫秒定时器的量化误差。
        body = [f"scoreboard players set lab_x{j} int {round(v * 10000)}" for j, v in enumerate(x)]
        body += [f"function lab:k/loop_{name}_{mode}"] * 8 + ["stopwatch restart lab:kernels"]
        body += [f"function lab:k/loop_{name}_{mode}"] * 128
        body += [
            f"data modify storage lab:io timing set value {{id:{i}}}",
            "execute store result storage lab:io timing.us int 1 run stopwatch query lab:kernels 1000000",
            *[
                f"execute store result storage lab:io timing.y{k} int 1 run scoreboard players get lab_y{k} int"
                for k in range(expected_outputs[name]["formula"].shape[1])
            ],
            f"function lab:log/{name} with storage lab:io timing",
            f"schedule function lab:trial/{i + 1} 1t" if i + 1 < len(plan) else "say VVE_DONE",
        ]
        function(entries, f"lab:trial/{i}", body)
    dest, lines = run(work, experiment, entries, len(plan) + 20)
    times = records(lines, "KERNEL")
    if len(times) != len(plan) or any(v[0] != i for i, v in enumerate(times)):
        raise ValueError("数值核计时缺失")
    for i, (name, r, mode, _) in enumerate(plan):
        predicted = np.floor(expected_outputs[name][mode][r] * 10000)
        actual = np.array(times[i][2:])
        if predicted.shape != actual.shape or np.max(abs(predicted - actual)) > 2 or times[i][1] <= 0:
            raise ValueError(f"{name}/{mode} 第 {r} 轮输出或计时无效：{actual}，期望 {predicted}")
        summary[name]["max_game_units"] = max(
            summary[name]["max_game_units"], int(np.max(abs(predicted - actual)))
        )
    for name in names:
        modes = {
            m: [times[i][1] / 4096 for i, (n, r, t, x) in enumerate(plan) if n == name and t == m]
            for m in ("formula", "inline", "cached", "small")
        }
        summary[name]["median_us"] = {k: float(np.median(v)) for k, v in modes.items()}
        summary[name]["paired_us"] = modes
    (dest / "screen.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(dest / "screen.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--fused", action="store_true", help="整段姿态和碰撞点的多输出试验")
    args = p.parse_args()
    screen(args.directory.resolve(), args.fused)
