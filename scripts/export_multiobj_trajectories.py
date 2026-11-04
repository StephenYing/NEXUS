from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.dataset_multi_object import MultiObjectDataset, resolve_checkpoint_inferred_graph_config, resolve_inferred_graph_config
from nexus.eval_multi_object_rollout import build_eval_indices, horizon_eff
from nexus.model_multi_object import KNN_EDGE_CONVENTION, infer_encoder_type_from_state_dict, validate_checkpoint_knn_edge_convention
from nexus.train_multi_object import (
    build_argparser as train_argparser,
    make_dynamics,
    model_kwargs_for_step,
    to_device,
)


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


@torch.no_grad()
def rollout_full(
    batch: Dict[str, Any],
    model: torch.nn.Module,
    n_substeps: int,
    force_clip: float,
    acc_clip: float,
    max_h: int,
    require_full_horizon: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_cur, v_cur = batch["x"], batch["v"]
    available = int(batch["rollout_x_next"].shape[0])
    he = horizon_eff(max_h, available, require_full=require_full_horizon)
    pred_x: List[torch.Tensor] = []
    pred_v: List[torch.Tensor] = []
    for s in range(he):
        kw = model_kwargs_for_step(batch, s)
        xp, vp = model(
            x_cur,
            v_cur,
            batch["types"],
            batch["dt"],
            batch["edge_index_struct"],
            batch["rest_lengths"],
            n_substeps=n_substeps,
            force_clip=force_clip,
            acc_clip=acc_clip,
            **kw,
        )
        pred_x.append(xp.detach().cpu())
        pred_v.append(vp.detach().cpu())
        x_cur, v_cur = xp.detach(), vp.detach()
    return torch.stack(pred_x, dim=0), torch.stack(pred_v, dim=0)


def export_one_ckpt(ckpt: str, data: str, out_dir: str, args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")

    model, cfg = load_model(ckpt, device)
    graph_config = resolve_inferred_graph_config(cfg)
    nobj = int(cfg.get("n_objects", args.n_objects))

    ds = MultiObjectDataset(
        data=data,
        mode=args.split,
        split=args.train_split,
        split_by=args.split_by,
        n_objects=nobj,
        rollout_steps=args.horizon,
        clip_rollout_to_available=not bool(getattr(args, "require_full_horizon", True)),
        min_stored_states=49 if getattr(args, "require_full_horizon", True) else 0,
        max_files=args.max_files,
        load_workers=args.file_load_workers,
        worker_cache_size=1,
        **graph_config,
    )

    indices = build_eval_indices(ds, max_per_k_tuple=args.max_per_k_tuple)
    os.makedirs(out_dir, exist_ok=True)
    exp_name = os.path.basename(os.path.dirname(ckpt))
    saved = 0

    for j, idx in enumerate(indices):
        b = to_device(ds[idx], device)
        pred_x, pred_v = rollout_full(
            b,
            model,
            n_substeps=args.n_substeps,
            force_clip=args.force_clip,
            acc_clip=args.acc_clip,
            max_h=args.horizon,
            require_full_horizon=bool(getattr(args, "require_full_horizon", True)),
        )
        he = int(pred_x.shape[0])
        gt_x = b["rollout_x_next"][:he].detach().cpu()
        gt_v = b["rollout_v_next"][:he].detach().cpu()
        k_values = b["k_values"].detach().cpu().numpy().reshape(-1)

        file_name = (
            f"ep_{j:05d}_fi{int(b['file_index'].detach().cpu()):03d}"
            f"_ei{int(b['episode_index'].detach().cpu()):04d}"
            f"_gi{int(b['group_index'].detach().cpu()):03d}"
            f"_ri{int(b['repeat_index'].detach().cpu()):03d}.npz"
        )
        np.savez_compressed(
            os.path.join(out_dir, file_name),
            exp_name=exp_name,
            ckpt=ckpt,
            n_objects=int(b["n_objects"].detach().cpu()),
            particle_radius=float(b["particle_radius"].detach().cpu()),
            dt=float(b["dt"].detach().cpu()),
            k_values=k_values,
            object_id=b["object_id"].detach().cpu().numpy(),
            types=b["types"].detach().cpu().numpy(),
            x0=b["x"].detach().cpu().numpy(),
            v0=b["v"].detach().cpu().numpy(),
            pred_x=pred_x.numpy(),
            pred_v=pred_v.numpy(),
            gt_x=gt_x.numpy(),
            gt_v=gt_v.numpy(),
            knn_edge_convention=KNN_EDGE_CONVENTION,
            checkpoint_knn_edge_convention=cfg["checkpoint_knn_edge_convention"],
            inferred_graph_config_json=json.dumps(graph_config, sort_keys=True),
            requested_horizon=int(args.horizon),
            effective_horizon=he,
            num_exported_states=he + 1,
        )
        saved += 1

    print(f"[done] ckpt={ckpt}")
    print(f"[done] out_dir={out_dir}")
    print(f"[done] saved_episodes={saved}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export closed-loop trajectories (pred/gt) for multi-object models.")
    p.add_argument("--data", required=True)
    p.add_argument("--n_objects", type=int, default=0, choices=[0, 2, 3, 5])
    p.add_argument("--ckpt", default="")
    p.add_argument("--ckpt_glob", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--split", default="valid", choices=["train", "valid", "val", "all"])
    p.add_argument("--train_split", type=float, default=0.9)
    p.add_argument("--split_by", default="group_index")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--file_load_workers", type=int, default=2)
    p.add_argument("--max_per_k_tuple", type=int, default=8)
    p.add_argument("--horizon", type=int, default=48)
    horizon_mode = p.add_mutually_exclusive_group()
    horizon_mode.add_argument("--require_full_horizon", dest="require_full_horizon", action="store_true")
    horizon_mode.add_argument("--allow_horizon_clipping", dest="require_full_horizon", action="store_false",
                              help="Explicit legacy-only mode; record the effective shortened horizon")
    p.set_defaults(require_full_horizon=True)
    p.add_argument("--n_substeps", type=int, default=4)
    p.add_argument("--force_clip", type=float, default=0.0)
    p.add_argument("--acc_clip", type=float, default=100.0)
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
        export_one_ckpt(ck, args.data, sub, args)


if __name__ == "__main__":
    main()
