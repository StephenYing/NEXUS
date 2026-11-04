from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from .dataset_multi_object import MultiObjectDataset, identity_collate, resolve_inferred_graph_config
from .model_multi_object import KNN_EDGE_CONVENTION, make_model

SAVE_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints_multi_object")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def parse_schedule(s: str, default: int) -> List[Tuple[int, int]]:
    if not s:
        return [(0, int(default))]
    out: List[Tuple[int, int]] = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            a, b = part.split(':', 1)
            out.append((int(a), int(b)))
        else:
            out.append((0, int(part)))
    out = sorted(out, key=lambda x: x[0])
    return out or [(0, int(default))]


def rollout_steps_for_epoch(args: argparse.Namespace, epoch: int) -> int:
    steps = int(args.rollout_steps)
    for e, h in parse_schedule(args.rollout_schedule, args.rollout_steps):
        if epoch >= e:
            steps = int(h)
    return max(1, steps)


def rollout_dataset_steps(args: argparse.Namespace) -> int:
    max_h = max([h for _, h in parse_schedule(args.rollout_schedule, args.rollout_steps)] + [int(args.rollout_steps), int(args.val_rollout_steps)])
    return max(1, int(max_h))


def normalized_loss(diff: torch.Tensor, scale: float, loss_type: str = "huber", huber_delta: float = 1.0) -> torch.Tensor:
    z = diff / max(float(scale), 1e-8)
    if loss_type == "mse":
        return torch.mean(z.pow(2))
    if loss_type == "l1":
        return torch.mean(torch.abs(z))
    return F.smooth_l1_loss(z, torch.zeros_like(z), beta=float(huber_delta), reduction="mean")


def inferred_graph_strain_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    edge_index: torch.Tensor,
    rest: torch.Tensor,
    *,
    loss_type: str = "huber",
    huber_delta: float = 0.25,
    max_edges: int = 0,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    row, col = edge_index[0], edge_index[1]
    if max_edges and int(max_edges) > 0 and row.numel() > int(max_edges):
        # deterministic striding, no stochastic leak
        idx = torch.linspace(0, row.numel() - 1, steps=int(max_edges), device=row.device).long()
        row, col = row[idx], col[idx]
        rest = rest[idx]
    rest = rest.to(device=pred.device, dtype=pred.dtype).reshape(-1).clamp_min(1e-8)
    ep = (torch.norm(pred[row] - pred[col], dim=-1) - rest) / rest
    eg = (torch.norm(gt[row] - gt[col], dim=-1) - rest) / rest
    if loss_type == "mse":
        return torch.mean((ep - eg.detach()).pow(2))
    if loss_type == "l1":
        return torch.mean(torch.abs(ep - eg.detach()))
    return F.smooth_l1_loss(ep, eg.detach(), beta=float(huber_delta), reduction="mean")


def model_kwargs_for_step(batch: Dict[str, Any], s: int) -> Dict[str, Any]:
    return {
        "x_init": batch["x_init"],
        "object_id": batch["object_id"],
        "edge_coef_k1": batch.get("edge_coef_k1", None),
        "edge_object_id": batch.get("edge_object_id", None),
        "k_triplet": batch.get("k_values", batch.get("k_triplet", batch.get("k_pair", batch.get("k", None)))),
        "vol": batch.get("vol", None),
        "particle_radius": batch.get("particle_radius", None),
        "env_normals": batch.get("env_normals", None),
        "gravity": batch.get("gravity", None),
        "gravity_on": batch.get("gravity_on", None),
        "extra_force": batch["rollout_extra_force"][s] if "rollout_extra_force" in batch else batch.get("extra_force", None),
        "extra_mask": batch["rollout_extra_mask"][s] if "rollout_extra_mask" in batch else batch.get("extra_mask", None),
        "gravity_force": batch["rollout_gravity_force"][s] if "rollout_gravity_force" in batch else batch.get("gravity_force", None),
        "gravity_mask": batch["rollout_gravity_mask"][s] if "rollout_gravity_mask" in batch else batch.get("gravity_mask", None),
        "total_force": batch["rollout_total_force"][s] if "rollout_total_force" in batch else batch.get("total_force", None),
        "total_mask": batch["rollout_total_mask"][s] if "rollout_total_mask" in batch else batch.get("total_mask", None),
        "ext_force": batch["rollout_ext_force"][s] if "rollout_ext_force" in batch else batch.get("ext_force", None),
        "ext_mask": batch["rollout_ext_mask"][s] if "rollout_ext_mask" in batch else batch.get("ext_mask", None),
    }


def sample_weight_from_k_values(batch: Dict[str, Any], k_ref: float, k_loss_power: float, k_loss_max: float) -> torch.Tensor:
    if float(k_loss_power) == 0.0:
        return torch.tensor(1.0, device=batch["x"].device)
    kp = batch.get("k_values", batch.get("k_triplet", batch.get("k_pair", batch.get("k", torch.tensor([k_ref], device=batch["x"].device))))).float().reshape(-1)
    kval = torch.max(kp) if kp.numel() >= 2 else kp.reshape(-1)[0]
    w = (kval / max(float(k_ref), 1e-8)).clamp_min(1e-8).pow(float(k_loss_power))
    if k_loss_max and k_loss_max > 0:
        w = w.clamp(max=float(k_loss_max))
    return w


def compute_rollout_loss(batch: Dict[str, Any], model: torch.nn.Module, scales: Dict[str, float], args: argparse.Namespace, rollout_steps: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    x_cur, v_cur = batch["x"], batch["v"]
    types = batch["types"]
    dyn = types == 0
    available = int(batch.get("rollout_steps", torch.tensor(1, device=x_cur.device)).detach().cpu().item())
    h = max(1, min(int(rollout_steps), available, int(batch["rollout_x_next"].shape[0])))
    if model.training and (args.state_noise_x > 0 or args.state_noise_v > 0):
        mask = dyn.view(-1, 1)
        if args.state_noise_x > 0:
            x_cur = x_cur + mask * torch.randn_like(x_cur) * float(args.state_noise_x)
        if args.state_noise_v > 0:
            v_cur = v_cur + mask * torch.randn_like(v_cur) * float(args.state_noise_v)
    edge_index = batch["edge_index_struct"]
    rest = batch["rest_lengths"]
    dt = batch["dt"]
    loss = torch.zeros((), device=x_cur.device)
    logs = {"loss_x": 0.0, "loss_v": 0.0, "loss_strain": 0.0}
    weight_sum = 0.0
    for s in range(h):
        kw = model_kwargs_for_step(batch, s)
        xp, vp = model(
            x_cur, v_cur,
            types,
            dt,
            edge_index,
            rest,
            n_substeps=args.n_substeps,
            force_clip=args.force_clip,
            acc_clip=args.acc_clip,
            **kw,
        )
        x_gt = batch["rollout_x_next"][s]
        v_gt = batch["rollout_v_next"][s]
        lx = normalized_loss(xp[dyn] - x_gt[dyn], scales["x"], args.loss_type, args.huber_delta)
        lv = normalized_loss(vp[dyn] - v_gt[dyn], scales["v"], args.loss_type, args.huber_delta)
        if args.rollout_strain_weight != 0:
            ls = inferred_graph_strain_loss(xp, x_gt, edge_index, rest, loss_type=args.strain_loss_type, huber_delta=args.strain_huber_delta, max_edges=args.strain_max_edges)
        else:
            ls = torch.zeros((), device=xp.device, dtype=xp.dtype)
        w = float(args.rollout_decay) ** float(s)
        loss = loss + w * (args.rollout_x_weight * lx + args.rollout_v_weight * lv + args.rollout_strain_weight * ls)
        logs["loss_x"] += float(lx.detach().cpu()) * w
        logs["loss_v"] += float(lv.detach().cpu()) * w
        logs["loss_strain"] += float(ls.detach().cpu()) * w
        weight_sum += w
        if args.detach_rollout_state:
            x_cur, v_cur = xp.detach(), vp.detach()
        else:
            x_cur, v_cur = xp, vp
    loss = loss / max(weight_sum, 1e-8)
    sw = sample_weight_from_k_values(batch, args.k_ref, args.k_loss_power, args.k_loss_max)
    loss = loss * sw
    logs = {k: v / max(weight_sum, 1e-8) for k, v in logs.items()}
    logs.update({"loss": float(loss.detach().cpu()), "rollout_steps_used": float(h), "sample_weight": float(sw.detach().cpu())})
    return loss, logs


def make_dataset(args: argparse.Namespace, mode: str) -> MultiObjectDataset:
    graph_config = resolve_inferred_graph_config(args)
    return MultiObjectDataset(
        min_stored_states=int(getattr(args, "min_stored_states", 49)),
        data=args.data,
        mode=mode,
        split=args.split,
        split_by=args.split_by,
        dt=args.dt,
        particle_radius=args.particle_radius,
        min_k=args.min_k,
        max_k=args.max_k,
        max_windows=args.max_windows_train if mode == "train" else args.max_windows_valid,
        max_files=args.max_files,
        n_objects=args.n_objects,
        load_workers=args.file_load_workers,
        worker_cache_size=args.worker_cache_size,
        rollout_steps=args.rollout_dataset_steps,
        oversample_contact=args.oversample_contact if mode == "train" else 1,
        contact_window_radius=args.contact_window_radius,
        contact_oversample_mode=args.contact_oversample_mode if mode == "train" else "none",
        **graph_config,
    )


def make_dynamics(args: argparse.Namespace, default_particle_radius: float) -> torch.nn.Module:
    return make_model(
        model_family=args.model_family,
        n_objects=args.n_objects,
        hidden=args.hidden,
        mlp_depth=args.mlp_depth,
        encoder_type=args.encoder_type,
        knn_backend=getattr(args, "knn_backend", "auto"),
        use_global_context=args.use_global_context,
        k_knn=args.k_knn,
        k_pair_contact=args.k_pair_contact,
        k_floor_contact=args.k_floor_contact,
        default_particle_radius=default_particle_radius,
        k_ref=args.k_ref,
        logk_center=args.logk_center,
        logk_scale=args.logk_scale,
        coef_detach=not args.disable_coef_detach,
        struct_mode=args.struct_mode,
        struct_scale=args.struct_scale,
        struct_gain_range=args.struct_gain_range,
        use_edge_coef=not args.disable_edge_coef,
        struct_k_power=args.struct_k_power,
        struct_geom_power=args.struct_geom_power,
        struct_min_gain=args.struct_min_gain,
        struct_geom_norm=args.struct_geom_norm,
        use_struct_damping=not args.disable_struct_damping,
        struct_damping_ratio=args.struct_damping_ratio,
        struct_damping_scale=args.struct_damping_scale,
        use_pair_contact=not args.disable_pair_contact,
        use_floor_contact=not args.disable_floor_contact,
        pair_contact_scale=args.pair_contact_scale,
        floor_contact_scale=args.floor_contact_scale,
        pair_contact_margin_factor=args.pair_contact_margin_factor,
        floor_contact_margin_factor=args.floor_contact_margin_factor,
        contact_edge_norm=args.contact_edge_norm,
        contact_barrier=args.contact_barrier,
        contact_softplus_beta=args.contact_softplus_beta,
        contact_cutoff_factor=args.contact_cutoff_factor,
        use_contact_damping=not args.disable_contact_damping,
        pair_contact_damping_scale=args.pair_contact_damping_scale,
        floor_contact_damping_scale=args.floor_contact_damping_scale,
        contact_damping_k_power=args.contact_damping_k_power,
        damping_param=args.damping_param,
        external_source=args.external_source,
        external_force_scale=args.external_force_scale,
        integrator=args.integrator,
        gravity=(0.0, args.gravity_y, 0.0),
    )


def estimate_scales(ds: MultiObjectDataset, max_samples: int = 64) -> Dict[str, float]:
    n = min(len(ds), max(1, int(max_samples)))
    xs, vs = [], []
    for idx in np.linspace(0, len(ds) - 1, num=n, dtype=int):
        b = ds[int(idx)]
        dyn = b["types"] == 0
        xs.append((b["x_next"][dyn] - b["x"][dyn]).reshape(-1))
        vs.append((b["v_next"][dyn] - b["v"][dyn]).reshape(-1))
    def rms(vals: Iterable[torch.Tensor], floor: float) -> float:
        z = torch.cat([v.float() for v in vals], dim=0)
        return max(float(torch.sqrt(torch.mean(z.pow(2))).item()), floor)
    return {"x": rms(xs, 1e-4), "v": rms(vs, 1e-4)}


def configure_runtime(args: argparse.Namespace) -> None:
    if args.cpu_threads and args.cpu_threads > 0:
        torch.set_num_threads(int(args.cpu_threads))
        try:
            torch.set_num_interop_threads(max(1, int(args.cpu_threads) // 2))
        except RuntimeError:
            pass
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    except Exception:
        pass

def make_loader(ds: MultiObjectDataset, args: argparse.Namespace, train: bool) -> DataLoader:
    if train and args.epoch_train_samples > 0:
        weights = torch.as_tensor(ds.sampler_weights(), dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=int(args.epoch_train_samples), replacement=True)
        shuffle = False
    elif (not train) and args.epoch_valid_samples > 0:
        sampler = RandomSampler(ds, replacement=True, num_samples=int(args.epoch_valid_samples))
        shuffle = False
    else:
        sampler = None
        shuffle = train
    nw = args.num_workers_train if train else args.num_workers_valid
    kwargs = dict(batch_size=1, sampler=sampler, shuffle=shuffle if sampler is None else False, collate_fn=identity_collate, num_workers=int(nw), pin_memory=True)
    if int(nw) > 0:
        kwargs.update(dict(persistent_workers=bool(args.persistent_workers), prefetch_factor=int(args.prefetch_factor)))
    return DataLoader(ds, **kwargs)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, scales: Dict[str, float], args: argparse.Namespace, device: torch.device, h: int) -> Dict[str, float]:
    model.eval()
    vals: Dict[str, List[float]] = {"loss": [], "loss_x": [], "loss_v": [], "loss_strain": []}
    for b in loader:
        b = to_device(b, device)
        loss, logs = compute_rollout_loss(b, model, scales, args, h)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite validation loss: {logs}")
        for k in vals:
            vals[k].append(float(logs.get(k, 0.0)))
    return {k: float(np.mean(v)) if v else float("inf") for k, v in vals.items()}


def train(args: argparse.Namespace) -> str:
    seed_everything(args.seed)
    configure_runtime(args)
    inferred_graph_config = resolve_inferred_graph_config(args)
    for key, value in inferred_graph_config.items():
        setattr(args, key, value)
    args.rollout_dataset_steps = rollout_dataset_steps(args)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")

    run_name = args.exp_name or "nexus_multiobj"
    run_dir = os.path.join(SAVE_ROOT, f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    t0 = time.time()
    ds_train = make_dataset(args, "train")
    ds_valid = make_dataset(args, "valid")
    print("[dataset/train]", json.dumps(ds_train.summary(), indent=2), flush=True)
    print("[dataset/valid]", json.dumps(ds_valid.summary(), indent=2), flush=True)
    print(f"[timing] dataset_init_seconds={time.time()-t0:.2f}", flush=True)

    scales = estimate_scales(ds_train, args.scale_samples)
    print("[scales]", json.dumps(scales, indent=2), flush=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "knn_edge_convention": KNN_EDGE_CONVENTION, "inferred_graph_config": inferred_graph_config, "scales": scales, "run_dir": run_dir}, f, indent=2)

    train_loader = make_loader(ds_train, args, train=True)
    valid_loader = make_loader(ds_valid, args, train=False)
    default_pr = ds_train.files[0].particle_radius if ds_train.files else 0.04
    model = make_dynamics(args, default_particle_radius=default_pr).to(device)
    print(f"[model] encoder_type_actual={getattr(model, 'encoder_type_actual', 'unknown')}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        h = rollout_steps_for_epoch(args, epoch)
        tr_logs: Dict[str, List[float]] = {"loss": [], "loss_x": [], "loss_v": [], "loss_strain": [], "rollout_steps_used": []}
        for b in train_loader:
            b = to_device(b, device)
            opt.zero_grad(set_to_none=True)
            loss, logs = compute_rollout_loss(b, model, scales, args, h)
            if not torch.isfinite(loss):
                print(f"[error] non-finite loss epoch={epoch} logs={logs}", flush=True)
                raise FloatingPointError("non-finite loss")
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            for k in tr_logs:
                tr_logs[k].append(float(logs.get(k, 0.0)))
        tr_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in tr_logs.items()}
        val: Optional[Dict[str, float]] = None
        if epoch % args.val_every == 0:
            val = validate(model, valid_loader, scales, args, device, args.val_rollout_steps)
            score = val["loss"]
            print(f"[epoch {epoch:04d}] h={h} train={json.dumps(tr_mean)} valid={json.dumps(val)}", flush=True)
            if score < best:
                best = score
                best_epoch = epoch
                state_best = {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "encoder_type_actual": getattr(model, "encoder_type_actual", args.encoder_type),
                    "knn_edge_convention": KNN_EDGE_CONVENTION,
                    "inferred_graph_config": inferred_graph_config,
                    "scales": scales,
                    "epoch": epoch,
                    "best_val": best,
                    "train_metrics": tr_mean,
                    "valid_metrics": val,
                }
                torch.save(state_best, os.path.join(run_dir, "best.pth"))
                print(f"[best] epoch={epoch} val={best:.6e}", flush=True)
        else:
            print(f"[epoch {epoch:04d}] h={h} train={json.dumps(tr_mean)}", flush=True)
        state = {
            "model": model.state_dict(),
            "args": vars(args),
            "encoder_type_actual": getattr(model, "encoder_type_actual", args.encoder_type),
            "knn_edge_convention": KNN_EDGE_CONVENTION,
            "inferred_graph_config": inferred_graph_config,
            "scales": scales,
            "epoch": epoch,
            "best_val": best,
            "train_metrics": tr_mean,
            "valid_metrics": val,
        }
        torch.save(state, os.path.join(run_dir, "last.pth"))
        torch.save(state, os.path.join(run_dir, f"epoch_{epoch:04d}.pth"))
    print(f"[done] run_dir={run_dir} best={best:.6e} best_epoch={best_epoch}", flush=True)
    return run_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NEXUS multi-object closed-loop rollout training.")
    p.add_argument("--data", required=True)
    p.add_argument("--min_stored_states", type=int, default=49,
                   help="Require 49-state true-H48 data; lower explicitly only for historical short-data runs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--n_objects", type=int, default=3, choices=[2,3,5])
    p.add_argument("--model_family", default="nexus", choices=["nexus"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--mlp_depth", type=int, default=2)
    p.add_argument("--encoder_type", default="auto", choices=["auto", "simple", "point_transformer", "pt"])
    p.add_argument("--knn_backend", default="auto", choices=["auto", "torch_cdist", "pyg"])
    p.add_argument("--use_global_context", action="store_true")
    p.add_argument("--split", type=float, default=0.9)
    p.add_argument("--split_by", default="group_index", choices=["group_index", "episode"])
    p.add_argument("--dt", type=float, default=0.0)
    p.add_argument("--particle_radius", type=float, default=0.0)
    p.add_argument("--min_k", type=float, default=0.0)
    p.add_argument("--max_k", type=float, default=0.0)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_windows_train", type=int, default=0)
    p.add_argument("--max_windows_valid", type=int, default=0)
    p.add_argument("--file_load_workers", type=int, default=2)
    p.add_argument("--worker_cache_size", type=int, default=1)
    p.add_argument("--num_workers_train", type=int, default=3)
    p.add_argument("--num_workers_valid", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--cpu_threads", type=int, default=2)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--epoch_train_samples", type=int, default=3072)
    p.add_argument("--epoch_valid_samples", type=int, default=256)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--scale_samples", type=int, default=64)
    p.add_argument("--rollout_steps", type=int, default=6)
    p.add_argument("--rollout_schedule", default="0:6,8:12,30:24")
    p.add_argument("--val_rollout_steps", type=int, default=24)
    p.add_argument("--rollout_decay", type=float, default=0.97)
    p.add_argument("--rollout_x_weight", type=float, default=1.0)
    p.add_argument("--rollout_v_weight", type=float, default=0.25)
    p.add_argument("--rollout_strain_weight", type=float, default=0.10)
    p.add_argument("--strain_loss_type", default="huber", choices=["huber", "l1", "mse"])
    p.add_argument("--strain_huber_delta", type=float, default=0.25)
    p.add_argument("--strain_max_edges", type=int, default=0)
    p.add_argument("--detach_rollout_state", action="store_true")
    p.add_argument("--state_noise_x", type=float, default=0.0)
    p.add_argument("--state_noise_v", type=float, default=0.0)
    p.add_argument("--loss_type", default="huber", choices=["huber", "l1", "mse"])
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--k_loss_power", type=float, default=0.0)
    p.add_argument("--k_loss_max", type=float, default=3.0)
    p.add_argument("--oversample_contact", type=int, default=6)
    p.add_argument("--contact_window_radius", type=int, default=6)
    p.add_argument("--contact_oversample_mode", default="metadata", choices=["metadata", "time", "none", "off"])
    p.add_argument("--k_obj", type=int, default=18)
    p.add_argument("--struct_graph_preset", default="solid")
    p.add_argument("--struct_add_mst", action="store_true")
    p.add_argument("--struct_long_k", type=int, default=8)
    p.add_argument("--struct_long_start_rank", type=int, default=12)
    p.add_argument("--struct_long_end_rank", type=int, default=96)
    p.add_argument("--struct_long_strategy", default="even")
    p.add_argument("--struct_seed_base", type=int, default=2026)
    p.add_argument("--struct_seed_protocol", default="visible_hash_v1", choices=["visible_hash_v1", "legacy_index_v1"])
    p.add_argument("--edge_coef_mode", default="inv_rest")
    p.add_argument("--edge_coef_clip_min", type=float, default=0.0)
    p.add_argument("--edge_coef_clip_max", type=float, default=100.0)
    p.add_argument("--struct_rest_eps", type=float, default=1e-6)
    p.add_argument("--disable_edge_coef", action="store_true")
    p.add_argument("--k_knn", type=int, default=12)
    p.add_argument("--k_pair_contact", type=int, default=4)
    p.add_argument("--k_floor_contact", type=int, default=8)
    p.add_argument("--k_ref", type=float, default=100.0)
    p.add_argument("--logk_center", type=float, default=2.0)
    p.add_argument("--logk_scale", type=float, default=1.0)
    p.add_argument("--disable_coef_detach", action="store_true")
    p.add_argument("--struct_mode", default="learned_times_k")
    p.add_argument("--struct_scale", type=float, default=100.0)
    p.add_argument("--struct_gain_range", type=float, default=0.75)
    p.add_argument("--struct_k_power", type=float, default=1.0)
    p.add_argument("--struct_geom_power", type=float, default=1.0)
    p.add_argument("--struct_min_gain", type=float, default=0.0)
    p.add_argument("--struct_geom_norm", default="mean")
    p.add_argument("--disable_struct_damping", action="store_true")
    p.add_argument("--struct_damping_ratio", type=float, default=0.015)
    p.add_argument("--struct_damping_scale", type=float, default=1.0)
    p.add_argument("--disable_pair_contact", action="store_true")
    p.add_argument("--disable_floor_contact", action="store_true")
    p.add_argument("--pair_contact_scale", type=float, default=60000.0)
    p.add_argument("--floor_contact_scale", type=float, default=80000.0)
    p.add_argument("--pair_contact_margin_factor", type=float, default=1.0)
    p.add_argument("--floor_contact_margin_factor", type=float, default=1.0)
    p.add_argument("--contact_edge_norm", default="mean_k")
    p.add_argument("--contact_barrier", default="relu")
    p.add_argument("--contact_softplus_beta", type=float, default=20.0)
    p.add_argument("--contact_cutoff_factor", type=float, default=0.0)
    p.add_argument("--disable_contact_damping", action="store_true")
    p.add_argument("--pair_contact_damping_scale", type=float, default=150.0)
    p.add_argument("--floor_contact_damping_scale", type=float, default=150.0)
    p.add_argument("--contact_damping_k_power", type=float, default=0.0)
    p.add_argument("--damping_param", default="learned", choices=["learned", "constant"])
    p.add_argument("--external_source", default="separate")
    p.add_argument("--external_force_scale", type=float, default=1.0)
    p.add_argument("--gravity_y", type=float, default=-4.0)
    p.add_argument("--integrator", default="semi_implicit")
    p.add_argument("--n_substeps", type=int, default=4)
    p.add_argument("--force_clip", type=float, default=0.0)
    p.add_argument("--acc_clip", type=float, default=100.0)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--exp_name", default="nexus_multiobj")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if not args.detach_rollout_state:
        # The NEXUS rollout protocol uses detached closed-loop states.
        args.detach_rollout_state = True
    if not args.struct_add_mst:
        args.struct_add_mst = True
    train(args)


if __name__ == "__main__":
    main()
