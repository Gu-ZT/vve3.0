"""由原求解器的四元数标签拟合接触修正，按整条轨迹留出。"""
import argparse
import json
from pathlib import Path

import numpy as np

from samples import quat_product

TRAIN = {11,23,37,47,61,73}
VALID = {89,101}
HOLD = {113,127,139,151}
ARCHITECTURES = [(4,), (8,), (16,), (8,8)]
MIN_ALIGNMENT = .92


def features(rows):
    n = rows[:,6:9] / 10000
    axes = rows[:,9:18].reshape(-1,3,3) / 10000
    local = np.floor(np.einsum("nij,nj->ni", axes,n)*10000 + 1e-8) / 10000
    face = np.argmax(abs(local), axis=1)
    aligned = local[np.arange(len(rows)),face]
    direction = np.eye(3)[face] * np.where(aligned>=0,1,-1)[:,None]
    tangent = np.cross(direction, local)
    q = rows[:,18:22] / 10000
    out = rows[:,22:26] / 10000
    relative = quat_product(q * [-1,-1,-1,1],out)
    relative *= np.where(relative[:,3] < 0,-1,1)[:,None]
    weight = (tangent*tangent).sum(axis=1)
    label = (tangent * relative[:,:3]).sum(axis=1) / np.maximum(weight,1e-12)
    return abs(aligned), tangent, q, out, label, weight


def design(x, shape):
    # 固定第一层折点，使用教师输出最小二乘拟合最后一层；小样本时避免拟合静止帧噪声。
    knots = np.linspace(0.,1.,shape[0], endpoint=False)
    w = [np.ones((1,shape[0]))]
    b = [-knots]
    if len(shape)>1:
        w.append(np.eye(shape[0]))
        b.append(-np.linspace(0, .5/shape[0], shape[0]))
    for weight, bias in zip(w,b):
        x = np.maximum(0, x@weight+bias)
    return np.column_stack([np.ones(len(x)),x]), w, b


def predict(model,x):
    y = x.reshape(-1,1).astype(np.float32)
    for index,(w,b) in enumerate(zip(model["w"],model["b"])):
        y = y @ np.asarray(w, np.float32) + np.asarray(b, np.float32)
        if index+1<len(model["w"]):
            y = np.maximum(0,y)
    return y[:,0]


def rotation_error(q, target, tangent, gain):
    v = np.floor(tangent * gain[:,None] * 10000) / 10000
    w = np.floor(np.sqrt(np.maximum(0,1-(v*v).sum(axis=1)))*10000) / 10000
    result = np.floor(quat_product(q,np.column_stack([v,w])) * 10000 + 1e-8) / 10000
    dot = abs((result * target).sum(axis=1))
    scale = np.linalg.norm(result,axis=1)*np.linalg.norm(target,axis=1)
    return 2*np.degrees(np.arccos(np.clip(dot/scale,0,1)))


def train(sample_file, destination):
    with np.load(sample_file) as source:
        rows = source["rows"]
        groups = json.loads(str(source["groups"]))
    terrain = np.array([groups[str(g)][0] for g in rows[:,0]])
    seed = np.array([groups[str(g)][1] for g in rows[:,0]])
    align,tangent,q,out,label,weight = features(rows)
    domain = (align>=MIN_ALIGNMENT)&(align<=1.001)&(np.linalg.norm(q,axis=1)>.995)
    x = np.clip((1-align)/(1-MIN_ALIGNMENT),0,1)
    models = {}
    for family,code in (("vehicle",1),("object",2)):
        relevant = ((rows[:,2]==code)|(terrain=="oracle")) & domain
        training = relevant & np.isin(seed,list(TRAIN)) & (weight>1e-5) & (terrain!="water")
        valid = relevant & np.isin(seed,list(VALID)) & (terrain!="water")
        hold = relevant & (np.isin(seed,list(HOLD)) | (terrain=="water"))
        if not all(mask.any() for mask in (training,valid,hold,hold&(terrain!="oracle"))):
            raise ValueError(f"{family} 缺少独立训练或验证轨迹")
        trials = []
        for shape in ARCHITECTURES:
            matrix,w,b = design(x[:,None],shape)
            importance = np.sqrt(weight[training])
            coefs = np.linalg.lstsq(matrix[training]*importance[:,None], label[training]*importance,rcond=1e-6)[0]
            model={"shape":list(shape),"w":[a.tolist() for a in w]+[coefs[1:,None].tolist()],
                   "b":[a.tolist() for a in b]+[[coefs[0]]]}
            errors = rotation_error(q[valid],out[valid],tangent[valid],predict(model,x[valid]))
            model["validation_deg"] = float(errors.max())
            trials.append(model)
        # 达到精度后优先参数最少的结构，留出集只评估最终选中的模型。
        best = min(trials,key=lambda m:(m["validation_deg"]>.25, sum(np.size(v) for v in m["w"]) if m["validation_deg"]<=.25 else m["validation_deg"]))
        error = rotation_error(q[hold],out[hold],tangent[hold],predict(best,x[hold]))
        best["holdout_deg"] = float(error.max())
        best["counts"] = [int(m.sum()) for m in (training,valid,hold)]
        best["architectures"] = [{"shape":m["shape"],"validation_deg":m["validation_deg"]} for m in trials]
        models[family] = best
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(models,indent=2,allow_nan=False), encoding="utf-8")
    return models


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("samples",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    for family,m in train(a.samples,a.output).items():
        print(family, m["shape"], m["counts"], m["holdout_deg"])
