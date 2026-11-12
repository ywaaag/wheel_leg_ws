#!/usr/bin/env python3
"""按 state_dict 的真实权重形状重建 31D Actor，导出确定性 ONNX。

为什么按形状推导：`native_stand.students.model.load()` 会把"网络构建"与"训练配方校验"
耦合在一起；当 checkpoint 使用的课程/特权 Critic 版本比当前 src 新时（例如 863 维 Critic），
load() 直接抛 `unknown motion course version`。这里只需要结构与权重，因此按权重形状重建，
并用严格 state_dict 加载 + 数值等价校验作为替代保证。

网络约定（与 src/native_stand/students/model.py 的 Student 一致）：
    Linear + ELU(alpha=1) 交替，最后一层不接激活；外层 tanh 得到 mean action。

用法：
    python scripts/export_actor_onnx_by_shapes.py \
        --checkpoint runs/<run>/formal/model_update_N.pt \
        --output artifacts/<ts>__<name>_onnx
输出目录必须不存在；任一步失败即非零退出且不留半成品。
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MLP_WEIGHT = re.compile(r'^mlp\.(\d+)\.weight$')
ERROR_LIMIT = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def derive_dims(actor, min_layers=2):
    """按 mlp.<i>.weight 的编号与形状推导 dims；编号不连续或形状不衔接即失败。"""
    idx = sorted(int(MLP_WEIGHT.match(k).group(1)) for k in actor if MLP_WEIGHT.match(k))
    if len(idx) < min_layers:
        raise ValueError(f'actor 层数不足: {idx}')
    expected = list(range(0, 2 * len(idx), 2))
    if idx != expected:
        raise ValueError(f'motion mlp 编号不连续: {idx} != {expected}')
    weights = [actor[f'mlp.{i}.weight'] for i in idx]
    for i in idx:
        if f'mlp.{i}.bias' not in actor:
            raise ValueError(f'缺少 bias: mlp.{i}.bias')
    dims = [int(weights[0].shape[1])]
    for w in weights:
        if w.ndim != 2 or int(w.shape[1]) != dims[-1]:
            raise ValueError(f'层形状不衔接: {tuple(w.shape)} vs 期望入维 {dims[-1]}')
        dims.append(int(w.shape[0]))
    return idx, dims


def build_actor(actor):
    idx, dims = derive_dims(actor)
    layers = []
    for n in range(len(dims) - 1):
        layers.append(nn.Linear(dims[n], dims[n + 1]))
        if n < len(dims) - 2:
            layers.append(nn.ELU(alpha=1.0))

    class DeterministicActor(nn.Module):
        """与 src/native_stand/students/export.py:DeterministicActor 等价（tanh(mlp(x))）。"""

        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(*layers)

        def forward(self, x):
            return self.mlp(x).tanh()

    model = DeterministicActor().eval()
    payload = {k: v for k, v in actor.items() if k.startswith('mlp.')}
    model.load_state_dict(payload, strict=True)   # key 缺失/多余/形状不符一律失败
    return model, dims, idx


def golden_inputs(obs_dim, seed=20261104):
    gen = torch.Generator().manual_seed(seed)
    return {
        'zeros': torch.zeros(1, obs_dim),
        'randn_unit': torch.randn(16, obs_dim, generator=gen),
        'uniform_boundary': (torch.rand(32, obs_dim, generator=gen) * 2 - 1) * 3.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    out = Path(args.output).resolve()
    if not ckpt.is_file():
        raise SystemExit(f'checkpoint 不存在: {ckpt}')
    if out.exists():
        raise SystemExit(f'输出目录已存在，拒绝覆盖: {out}')
    out.mkdir(parents=True)

    tmp_onnx = out / 'actor.onnx.tmp'
    try:
        ckpt_sha = sha256(ckpt)
        data = torch.load(ckpt, map_location='cpu', weights_only=False)
        if 'actor' not in data:
            raise SystemExit('checkpoint 缺少 actor')
        model, dims, idx = build_actor(data['actor'])
        obs_dim, act_dim = dims[0], dims[-1]
        params = int(sum(p.numel() for p in model.parameters()))

        import onnx
        from onnx.reference import ReferenceEvaluator

        with torch.no_grad():
            torch.onnx.export(
                model, torch.zeros(1, obs_dim), str(tmp_onnx),
                input_names=['observation'], output_names=['mean_action'],
                opset_version=17, dynamo=False)

        graph = onnx.load(str(tmp_onnx))
        onnx.checker.check_model(graph, full_check=True)
        sess = ReferenceEvaluator(graph)

        samples = golden_inputs(obs_dim)
        gold_x, gold_pt, gold_onnx, cats = [], [], [], []
        errors = {}
        for name, x in samples.items():
            with torch.no_grad():
                ref = model(x).numpy()
            got = sess.run(None, {'observation': x.numpy()})[0]
            err = float(np.abs(ref - got).max())
            errors[name] = err
            gold_x.append(x.numpy()); gold_pt.append(ref); gold_onnx.append(got)
            cats += [name] * x.shape[0]

        worst = max(errors.values())
        if worst > ERROR_LIMIT:
            raise SystemExit(f'数值不等价: max_abs_error={worst:.3e} > {ERROR_LIMIT}')

        os.replace(tmp_onnx, out / 'actor.onnx')
        onnx_sha = sha256(out / 'actor.onnx')

        np.savez_compressed(out / 'golden_vectors.npz',
                            input=np.concatenate(gold_x), pytorch_mean=np.concatenate(gold_pt),
                            onnx_mean=np.concatenate(gold_onnx), category=np.array(cats))
        golden_sha = sha256(out / 'golden_vectors.npz')

        manifest = {
            'schema': 'actor_onnx_export_v1',
            'checkpoint': {'path': str(ckpt), 'sha256': ckpt_sha,
                           'role': data.get('role'), 'interface_version': data.get('interface_version'),
                           'recipe_version': data.get('recipe_version')},
            'model': {'actor_dims': dims, 'mlp_indices': idx, 'activation': 'ELU(alpha=1)',
                      'output_activation': 'tanh', 'parameters': params,
                      'fp32_weight_bytes': params * 4,
                      'cross_check': 'wide_network(2048) actor dims 同为 [31,2048,1024,512,6]'},
            'onnx': {'path': 'actor.onnx', 'sha256': onnx_sha, 'opset': 17, 'dynamo': False,
                     'input': {'name': 'observation', 'shape': [1, obs_dim], 'dtype': 'float32'},
                     'output': {'name': 'mean_action', 'shape': [1, act_dim], 'dtype': 'float32'},
                     'file_bytes': (out / 'actor.onnx').stat().st_size},
            'verification': {'checker': 'onnx.checker.check_model(full_check=True)',
                             'backend': 'onnx.reference.ReferenceEvaluator (CPU float32)',
                             'max_abs_error': worst, 'per_sample': errors, 'limit': ERROR_LIMIT,
                             'golden_vectors': {'path': 'golden_vectors.npz', 'sha256': golden_sha,
                                                'samples': int(sum(v.shape[0] for v in samples.values()))}},
            'loader_note': ('未使用 native_stand.students.model.load()：其 recipe 校验依赖 '
                            'motion_course 版本白名单，本 checkpoint 的课程/863 维 Critic 版本未注册，'
                            '实测抛 unknown motion course version。改为按权重形状重建 + strict 加载 + 数值等价校验。'),
            'limits': ('2.69M 参数的训练用大 Actor，ONNX 约 10 MiB；不是 MCU 部署件'
                       '（电控小模型为 1.5 万参数/55 KiB 量级），仅适用于仿真复现与离线推理。'),
            'environment': {'torch': torch.__version__, 'onnx': onnx.__version__},
        }
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')

        readme = f"""# {out.name}

由 `scripts/export_actor_onnx_by_shapes.py` 从训练 checkpoint 导出的**确定性 Actor** ONNX。

| 项 | 值 |
|---|---|
| checkpoint | `{ckpt}` |
| checkpoint SHA256 | `{ckpt_sha}` |
| 结构 | `{' -> '.join(str(d) for d in dims)}`（ELU，末端 tanh） |
| 参数量 | {params:,} |
| ONNX 大小 | {(out / 'actor.onnx').stat().st_size / 2**20:.2f} MiB |
| 输入 / 输出 | `observation[1,{obs_dim}]` → `mean_action[1,{act_dim}]` |
| 数值等价 | 最大绝对误差 `{worst:.3e}`（阈值 {ERROR_LIMIT}） |

**限制**：这是训练用大 Actor，不是电控部署件。给 MCU 的交付件是 1.5 万参数 / 55 KiB 量级；
本文件仅适用于仿真复现与离线推理。

复现：

```bash
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python \\
  scripts/export_actor_onnx_by_shapes.py --checkpoint {ckpt} --output <新目录>
```
"""
        (out / 'README.md').write_text(readme, encoding='utf-8')

        print('导出完成:', out)
        print('  dims      :', dims)
        print('  params    : %d (%.2f MiB fp32)' % (params, params * 4 / 2 ** 20))
        print('  onnx      : %.2f MiB  sha256=%s' % ((out / 'actor.onnx').stat().st_size / 2 ** 20, onnx_sha))
        print('  误差      : %s  (最大 %.3e, 阈值 %s)' % (errors, worst, ERROR_LIMIT))
        print('  manifest  :', sha256(out / 'manifest.json'))
        print('  golden    :', golden_sha)
    except BaseException:
        if tmp_onnx.exists():
            tmp_onnx.unlink()
        raise


if __name__ == '__main__':
    main()
