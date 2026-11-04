from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .dataset_multi_object import MultiObjectDataset, resolve_checkpoint_inferred_graph_config, resolve_inferred_graph_config
from .model_multi_object import KNN_EDGE_CONVENTION, infer_encoder_type_from_state_dict, validate_checkpoint_knn_edge_convention
from .train_multi_object import make_dynamics, to_device, model_kwargs_for_step, build_argparser as train_argparser


def horizon_eff(h: int, available: int, require_full: bool = True) -> int:
    requested, available = int(h), int(available)
    if requested < 1 or available < 1:
        raise ValueError("Horizon and available transitions must both be positive")
    if require_full and requested > available:
        raise ValueError(
            f"Requested {requested} transitions, but ground truth contains only {available}; "
            f"a true H={requested} rollout requires at least {requested + 1} stored states"
        )
    return min(requested, available)


def mean_obj_l2(a: torch.Tensor, b: torch.Tensor, dyn: torch.Tensor) -> float:
    return float(torch.mean(torch.norm((a - b)[dyn], dim=-1)).detach().cpu())


def strain_l1(a: torch.Tensor, b: torch.Tensor, edge_index: torch.Tensor, rest: torch.Tensor) -> float:
    if edge_index.numel() == 0:
        return 0.0
    row, col = edge_index[0], edge_index[1]
    rest = rest.to(device=a.device, dtype=a.dtype).reshape(-1).clamp_min(1e-8)
    ea = (torch.norm(a[row] - a[col], dim=-1) - rest) / rest
    eb = (torch.norm(b[row] - b[col], dim=-1) - rest) / rest
    return float(torch.mean(torch.abs(ea - eb)).detach().cpu())


def pair_penetration(x: torch.Tensor, types: torch.Tensor, object_id: torch.Tensor, pr: float, n_objects: int) -> Tuple[float, float]:
    pens = []
    for a in range(int(n_objects)):
        idx_a = torch.where((types == 0) & (object_id == a))[0]
        if idx_a.numel() == 0:
            continue
        for b in range(a + 1, int(n_objects)):
            idx_b = torch.where((types == 0) & (object_id == b))[0]
            if idx_b.numel() == 0:
                continue
            d = torch.cdist(x[idx_a], x[idx_b])
            pens.append(torch.clamp(2.0 * float(pr) - d, min=0.0))
    if not pens:
        return 0.0, 0.0
    mean_val = torch.stack([p.mean() for p in pens]).mean()
    max_val = torch.stack([p.max() for p in pens]).max()
    return float(mean_val.detach().cpu()), float(max_val.detach().cpu())

def floor_penetration(x: torch.Tensor, types: torch.Tensor, floor_height: float, pr: float) -> Tuple[float, float]:
    dyn = torch.where(types == 0)[0]
    if dyn.numel() == 0:
        return 0.0, 0.0
    pen = torch.clamp(float(floor_height) + float(pr) - x[dyn, 1], min=0.0)
    return float(pen.mean().detach().cpu()), float(pen.max().detach().cpu())


@torch.no_grad()
def rollout_episode(batch: Dict[str, Any], model: torch.nn.Module, args: argparse.Namespace, horizons: List[int]) -> Dict[str, Any]:
    x_cur, v_cur = batch["x"], batch["v"]
    types = batch["types"]
    object_id = batch["object_id"]
    dyn = types == 0
    available = int(batch["rollout_x_next"].shape[0])
    require_full = bool(getattr(args, "require_full_horizon", True))
    max_h = horizon_eff(max(horizons), available, require_full=require_full)
    pred_x: List[torch.Tensor] = []
    pred_v: List[torch.Tensor] = []
    for s in range(max_h):
        kw = model_kwargs_for_step(batch, s)
        xp, vp = model(
            x_cur, v_cur,
            types,
            batch["dt"],
            batch["edge_index_struct"],
            batch["rest_lengths"],
            n_substeps=args.n_substeps,
            force_clip=args.force_clip,
            acc_clip=args.acc_clip,
            **kw,
        )
        pred_x.append(xp.detach())
        pred_v.append(vp.detach())
        x_cur, v_cur = xp.detach(), vp.detach()
    pr = float(batch["particle_radius"].detach().cpu())
    fh = float(batch["floor_height"].detach().cpu()) if "floor_height" in batch else 0.0
    out: Dict[str, Any] = {}
    for h in horizons:
        he = horizon_eff(h, available, require_full=require_full)
        xs = pred_x[:he]
        vs = pred_v[:he]
        gx = batch["rollout_x_next"][:he]
        gv = batch["rollout_v_next"][:he]
        x_errors = [mean_obj_l2(xs[i], gx[i], dyn) for i in range(he)]
        v_errors = [mean_obj_l2(vs[i], gv[i], dyn) for i in range(he)]
        strain_errors = [strain_l1(xs[i], gx[i], batch["edge_index_struct"], batch["rest_lengths"]) for i in range(he)]
        nobj = int(batch.get("n_objects", torch.tensor(3, device=x_cur.device)).detach().cpu().item())
        pair_pens = [pair_penetration(xs[i], types, object_id, pr, nobj) for i in range(he)]
        floor_pens = [floor_penetration(xs[i], types, fh, pr) for i in range(he)]
        out[f"requested_h{h}"] = int(h)
        out[f"effective_h{h}"] = he
        out[f"x_exact_h{h}"] = x_errors[he - 1]
        out[f"x_mean_h{h}"] = float(np.mean(x_errors))
        out[f"v_exact_h{h}"] = v_errors[he - 1]
        out[f"v_mean_h{h}"] = float(np.mean(v_errors))
        out[f"strain_mean_h{h}"] = float(np.mean(strain_errors))
        out[f"pair_pen_mean_h{h}"] = float(np.mean([p[0] for p in pair_pens]))
        out[f"pair_pen_max_h{h}"] = float(np.max([p[1] for p in pair_pens]))
        out[f"floor_pen_mean_h{h}"] = float(np.mean([p[0] for p in floor_pens]))
        out[f"floor_pen_max_h{h}"] = float(np.max([p[1] for p in floor_pens]))
    return out


def build_eval_indices(ds: MultiObjectDataset, max_per_k_tuple: int = 0) -> List[int]:
    # Use windows at t=0 only. This is true closed-loop episode rollout.
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, (_, _, t) in enumerate(ds.windows.tolist()):
        if int(t) != 0:
            continue
        b = ds[idx]
        kp = b["k_values"].numpy().reshape(-1)
        key = ",".join([f"{x:g}" for x in kp])
        groups[key].append(idx)
    out: List[int] = []
    for key in sorted(groups.keys()):
        vals = groups[key]
        if max_per_k_tuple and max_per_k_tuple > 0:
            vals = vals[: int(max_per_k_tuple)]
        out.extend(vals)
    return out


def load_model(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("args", {})
    parser = train_argparser()
    defaults = vars(parser.parse_args(["--data", "DUMMY"]))
    merged = {**defaults, **cfg}
    merged["knn_backend"] = ckpt.get("knn_backend_actual", merged.get("knn_backend", "auto"))
    merged.update(resolve_checkpoint_inferred_graph_config(ckpt))
    has_versioned_knn = validate_checkpoint_knn_edge_convention(ckpt)
    merged["checkpoint_knn_edge_convention"] = ckpt.get("knn_edge_convention", "legacy_unknown")
    if not has_versioned_knn:
        warnings.warn(
            "Legacy checkpoint has no KNN edge convention metadata; SimplePointConv "
            "weights may require retraining under receiver_sender_v1.",
            RuntimeWarning,
            stacklevel=2,
        )
    encoder_type_actual = ckpt.get("encoder_type_actual") or infer_encoder_type_from_state_dict(ckpt["model"])
    if encoder_type_actual:
        merged["encoder_type"] = encoder_type_actual
    ns = argparse.Namespace(**merged)
    model = make_dynamics(ns, default_particle_radius=float(getattr(ns, "particle_radius", 0.04) or 0.04)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, merged

def eval_ckpt(ckpt: str, data: str, out_dir: str, args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    model, cfg = load_model(ckpt, device)
    graph_config = resolve_inferred_graph_config(cfg)
    max_h = max(args.horizons)
    ds = MultiObjectDataset(
        data=data,
        mode=args.split,
        split=args.train_split,
        split_by=args.split_by,
        n_objects=int(cfg.get("n_objects", args.n_objects)),
        rollout_steps=max_h,
        clip_rollout_to_available=not bool(getattr(args, "require_full_horizon", True)),
        min_stored_states=49 if getattr(args, "require_full_horizon", True) else 0,
        max_files=args.max_files,
        load_workers=args.file_load_workers,
        worker_cache_size=1,
        **graph_config,
    )
    if ds.rollout_steps != ds.requested_rollout_steps:
        print(
            f"[eval] requested H={ds.requested_rollout_steps}; "
            f"using the last {ds.rollout_steps} available transitions",
            flush=True,
        )
    indices = build_eval_indices(ds, max_per_k_tuple=args.max_per_k_tuple)
    rows: List[Dict[str, Any]] = []
    os.makedirs(out_dir, exist_ok=True)
    for idx in indices:
        b = to_device(ds[idx], device)
        m = rollout_episode(b, model, args, args.horizons)
        kp = b["k_values"].detach().cpu().numpy().reshape(-1)
        m.update({
            "ckpt": ckpt,
            "exp_name": os.path.basename(os.path.dirname(ckpt)),
            "dataset_index": int(b["dataset_index"].detach().cpu()),
            "file_index": int(b["file_index"].detach().cpu()),
            "episode_index": int(b["episode_index"].detach().cpu()),
            "group_index": int(b["group_index"].detach().cpu()),
            "repeat_index": int(b["repeat_index"].detach().cpu()),
            "n_objects": int(b["n_objects"].detach().cpu()),
            "k_tuple": ",".join([f"{x:g}" for x in kp]),
        })
        rows.append(m)
    if not rows:
        raise RuntimeError("No eval rows produced; check split and data")
    keys = sorted(rows[0].keys())
    ep_csv = os.path.join(out_dir, "multiobj_rollout_episode_rows.csv")
    with open(ep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, np.floating))]
    overall = {k: float(np.mean([float(r[k]) for r in rows])) for k in numeric_keys}
    overall.update({"num_episodes": len(rows), "ckpt": ckpt, "knn_edge_convention": KNN_EDGE_CONVENTION, "checkpoint_knn_edge_convention": cfg["checkpoint_knn_edge_convention"], "inferred_graph_config": graph_config})
    overall.update({
        "primary_position_metric": "x_exact",
        "requested_horizons": [int(h) for h in args.horizons],
        "horizon_policy": "require_full" if getattr(args, "require_full_horizon", True) else "clip_to_available",
        "stored_states_min": min(int(info.n_t) for info in ds.files),
        "stored_states_max": max(int(info.n_t) for info in ds.files),
    })
    with open(os.path.join(out_dir, "multiobj_rollout_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r["k_tuple"])].append(r)
    by_rows = []
    for key, vals in sorted(groups.items()):
        d = {"k_tuple": key, "num_episodes": len(vals)}
        for nk in numeric_keys:
            d[nk] = float(np.mean([float(v[nk]) for v in vals]))
        by_rows.append(d)
    by_csv = os.path.join(out_dir, "multiobj_rollout_by_ktuple.csv")
    with open(by_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(by_rows[0].keys()))
        w.writeheader()
        w.writerows(by_rows)
    print(json.dumps(overall, indent=2), flush=True)
    print(f"[saved] {ep_csv}\n[saved] {by_csv}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Closed-loop rollout eval for multi-object models")
    p.add_argument("--data", required=True)
    p.add_argument("--n_objects", type=int, default=0, choices=[0,2,3,5])
    p.add_argument("--ckpt", default="")
    p.add_argument("--ckpt_glob", default="")
    p.add_argument("--out", default="eval_multiobj_rollout")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--split", default="valid", choices=["train", "valid", "val", "all"])
    p.add_argument("--train_split", type=float, default=0.9)
    p.add_argument("--split_by", default="group_index")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--file_load_workers", type=int, default=2)
    p.add_argument("--max_per_k_tuple", type=int, default=50)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12, 24, 48])
    horizon_mode = p.add_mutually_exclusive_group()
    horizon_mode.add_argument("--require_full_horizon", dest="require_full_horizon", action="store_true",
                              help="Require full horizons (default); true H48 requires at least 49 stored states")
    horizon_mode.add_argument("--allow_horizon_clipping", dest="require_full_horizon", action="store_false",
                              help="Explicit legacy-only mode: clip to available transitions and record the effective horizon")
    p.set_defaults(require_full_horizon=True)
    p.add_argument("--n_substeps", type=int, default=4)
    p.add_argument("--force_clip", type=float, default=0.0)
    p.add_argument("--acc_clip", type=float, default=100.0)
    # graph defaults if checkpoint config does not contain them
    p.add_argument("--k_obj", type=int, default=18)
    p.add_argument("--struct_graph_preset", default="solid")
    p.add_argument("--struct_long_k", type=int, default=8)
    p.add_argument("--struct_long_start_rank", type=int, default=12)
    p.add_argument("--struct_long_end_rank", type=int, default=96)
    p.add_argument("--struct_long_strategy", default="even")
    p.add_argument("--edge_coef_mode", default="inv_rest")
    p.add_argument("--edge_coef_clip_max", type=float, default=100.0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    ckpts = []
    if args.ckpt:
        ckpts.append(args.ckpt)
    if args.ckpt_glob:
        ckpts.extend(sorted(glob.glob(args.ckpt_glob)))
    if not ckpts:
        raise ValueError("Need --ckpt or --ckpt_glob")
    for ck in ckpts:
        sub = os.path.join(args.out, os.path.basename(os.path.dirname(ck))) if len(ckpts) > 1 else args.out
        eval_ckpt(ck, args.data, sub, args)


if __name__ == "__main__":
    main()
