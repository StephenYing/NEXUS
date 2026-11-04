#!/usr/bin/env python3
"""
Generate multi-object spring-mass trajectories from PhysCtrl geometry.
Saved simulator graphs are for evaluation purposes only; NEXUS infers its own structural graphs.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm

try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    cKDTree = None
    _HAS_SCIPY = False

Array = np.ndarray



def parse_k_tuples(s: str, n_objects: int) -> List[Tuple[float, ...]]:
    vals: List[Tuple[float, ...]] = []
    for chunk in str(s).strip().split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        xs = [float(v.strip()) for v in chunk.replace(":", ",").split(",") if v.strip()]
        if len(xs) != int(n_objects):
            raise ValueError(f"Bad k tuple {chunk!r}; expected {n_objects} values")
        vals.append(tuple(xs))
    if not vals:
        raise ValueError("empty k tuple list")
    return vals


def default_k_tuples(n: int) -> str:
    if n == 2:
        return "10,10;10,100;100,10;20,200;200,20;50,50;100,100;500,100;100,500;500,500"
    if n == 3:
        return "10,10,10;10,100,100;100,10,100;100,100,10;10,100,500;500,100,10;50,50,50;100,100,100;500,100,100;100,500,100;100,100,500;500,500,500"
    if n == 5:
        return ";".join([
            "10,10,10,10,10",
            "50,50,50,50,50",
            "100,100,100,100,100",
            "500,500,500,500,500",
            "10,100,100,100,100",
            "100,10,100,100,100",
            "100,100,10,100,100",
            "100,100,100,10,100",
            "100,100,100,100,10",
            "500,100,100,100,10",
            "10,100,500,100,100",
            "100,500,100,10,100",
        ])
    rows: List[List[float]] = [[10.0] * n, [100.0] * n, [500.0] * n]
    for i in range(min(n, 6)):
        a = [100.0] * n; a[i] = 10.0; rows.append(a)
        b = [100.0] * n; b[i] = 500.0; rows.append(b)
    return ";".join(",".join(str(int(x)) for x in r) for r in rows)


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for item in str(s).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def pairwise_d2(x: Array, y: Optional[Array] = None) -> Array:
    x = np.asarray(x, dtype=np.float32)
    if y is None:
        y = x
    else:
        y = np.asarray(y, dtype=np.float32)
    xx = np.sum(x * x, axis=1, keepdims=True)
    yy = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(xx + yy - 2.0 * (x @ y.T), 0.0)


def farthest_point_sample(points: Array, n_out: int, seed: int) -> Tuple[Array, Array]:
    points = np.asarray(points, dtype=np.float32)
    n = int(points.shape[0])
    if n_out >= n:
        idx = np.arange(n, dtype=np.int64)
        return points.copy(), idx
    rng = np.random.RandomState(seed)
    sel = np.empty((n_out,), dtype=np.int64)
    sel[0] = int(rng.randint(0, n))
    d2 = np.sum((points - points[sel[0]]) ** 2, axis=1)
    for i in range(1, n_out):
        sel[i] = int(np.argmax(d2))
        d2 = np.minimum(d2, np.sum((points - points[sel[i]]) ** 2, axis=1))
    return points[sel].copy(), sel


def normalize_object(x: Array, normalize_diameter: float) -> Tuple[Array, float, Array]:
    x = np.asarray(x, dtype=np.float32)
    center = 0.5 * (x.min(axis=0) + x.max(axis=0))
    x0 = x - center[None]
    if normalize_diameter and normalize_diameter > 0:
        scale = float(np.max(np.ptp(x0, axis=0)))
        if scale < 1e-8:
            scale = 1.0
        x0 = x0 / scale * float(normalize_diameter)
    else:
        scale = 1.0
    return x0.astype(np.float32), scale, center.astype(np.float32)


def rotation_yaw(theta: float) -> Array:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def random_horizontal_unit(rng: np.random.RandomState) -> Array:
    a = float(rng.uniform(0, 2 * math.pi))
    return np.array([math.cos(a), 0.0, math.sin(a)], dtype=np.float32)


def unit(v: Array, fallback: Optional[Array] = None) -> Array:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.asarray(fallback if fallback is not None else [1.0, 0.0, 0.0], dtype=np.float32)
    return v / n


def read_h5_geometry(h5_path: str) -> Tuple[Array, Dict[str, object]]:
    meta: Dict[str, object] = {}
    with h5py.File(h5_path, "r") as f:
        if "x" not in f:
            raise KeyError(f"{h5_path} has no dataset 'x'")
        x = np.asarray(f["x"], dtype=np.float32)
        if x.ndim == 4:
            x0 = x[0, 0]
        elif x.ndim == 3:
            x0 = x[0]
        elif x.ndim == 2 and x.shape[-1] == 3:
            x0 = x
        else:
            raise ValueError(f"unsupported x shape in {h5_path}: {x.shape}")
        meta["E_physctrl"] = float(np.asarray(f["E"]).reshape(-1)[0]) if "E" in f else np.nan
        meta["nu_physctrl"] = float(np.asarray(f["nu"]).reshape(-1)[0]) if "nu" in f else np.nan
        meta["mat_type_physctrl"] = int(np.asarray(f["mat_type"]).reshape(-1)[0]) if "mat_type" in f else -1
    return x0.astype(np.float32), meta

# Hidden simulator graph. Saved only for diagnostics.


def canonical_edge(a: int, b: int) -> Tuple[int, int]:
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def symmetric_knn_edges(x: Array, k: int) -> Array:
    n = int(x.shape[0])
    if n <= 1 or k <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    k = min(int(k), n - 1)
    d2 = pairwise_d2(x)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    edges = set()
    for i in range(n):
        for j in nn[i]:
            a, b = canonical_edge(i, int(j))
            if a != b:
                edges.add((a, b))
    return np.asarray(sorted(edges), dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)


def prim_mst_edges(x: Array) -> Array:
    n = int(x.shape[0])
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    d2 = pairwise_d2(x)
    used = np.zeros(n, dtype=bool); used[0] = True
    min_d = d2[0].copy(); parent = np.zeros(n, dtype=np.int64)
    edges: List[Tuple[int, int]] = []
    for _ in range(n - 1):
        cand = np.where(~used, min_d, np.inf)
        j = int(np.argmin(cand))
        if not np.isfinite(cand[j]):
            break
        i = int(parent[j])
        edges.append(canonical_edge(i, j))
        used[j] = True
        better = d2[j] < min_d
        parent[better] = j
        min_d[better] = d2[j, better]
    return np.asarray(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)


def midrange_edges(x: Array, k_long: int, start_rank: int, end_rank: int, rng: np.random.RandomState, strategy: str) -> Array:
    n = int(x.shape[0])
    if n <= 1 or k_long <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    d2 = pairwise_d2(x)
    np.fill_diagonal(d2, np.inf)
    order = np.argsort(d2, axis=1)
    s = max(1, int(start_rank)); e = min(max(s + 1, int(end_rank)), n)
    edges = set(); strategy = str(strategy).lower()
    for i in range(n):
        pool = order[i, s:e]
        pool = pool[np.isfinite(d2[i, pool])]
        if pool.size == 0:
            continue
        kk = min(int(k_long), int(pool.size))
        if strategy == "random":
            chosen = rng.choice(pool, size=kk, replace=False)
        elif strategy == "even":
            if kk == 1:
                chosen = np.array([pool[len(pool)//2]], dtype=np.int64)
            else:
                chosen = pool[np.linspace(0, len(pool)-1, kk).round().astype(np.int64)]
        else:
            chosen = pool[:kk]
        for j in chosen:
            a, b = canonical_edge(i, int(j))
            if a != b:
                edges.add((a, b))
    return np.asarray(sorted(edges), dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)


def union_edges(*arrs: Array) -> Array:
    edges = set()
    for arr in arrs:
        arr = np.asarray(arr, dtype=np.int64)
        if arr.size == 0:
            continue
        for a, b in arr.reshape(-1, 2):
            u, v = canonical_edge(int(a), int(b))
            if u != v:
                edges.add((u, v))
    return np.asarray(sorted(edges), dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)


def graph_params(args: argparse.Namespace) -> Tuple[int, int, int, int]:
    preset = str(args.graph_preset).lower()
    if preset == "local":
        return int(args.graph_k), 0, int(args.long_start_rank), int(args.long_end_rank)
    if preset == "solid":
        return max(int(args.graph_k), 18), max(int(args.long_k), 8), int(args.long_start_rank), int(args.long_end_rank)
    if preset == "rigidish":
        return max(int(args.graph_k), 24), max(int(args.long_k), 16), int(args.long_start_rank), max(int(args.long_end_rank), 128)
    return int(args.graph_k), int(args.long_k), int(args.long_start_rank), int(args.long_end_rank)


def build_spring_graph(x_rest: Array, args: argparse.Namespace, rng: np.random.RandomState) -> Tuple[Array, Array]:
    gk, lk, sr, er = graph_params(args)
    e_knn = symmetric_knn_edges(x_rest, gk)
    e_mst = prim_mst_edges(x_rest) if bool(args.add_mst) else np.zeros((0, 2), dtype=np.int64)
    e_long = midrange_edges(x_rest, lk, sr, er, rng, str(args.long_strategy)) if lk > 0 else np.zeros((0, 2), dtype=np.int64)
    edges = union_edges(e_knn, e_mst, e_long)
    if edges.size == 0:
        return edges, np.zeros((0,), dtype=np.float32)
    rest = np.linalg.norm(x_rest[edges[:, 0]] - x_rest[edges[:, 1]], axis=1).astype(np.float32)
    keep = rest > 1e-8
    return edges[keep].astype(np.int64), rest[keep].astype(np.float32)


def spring_coefficients(rest: Array, k: float, args: argparse.Namespace) -> Array:
    rest = np.asarray(rest, dtype=np.float32)
    eps = float(args.spring_rest_eps)
    if str(args.spring_energy_mode) == "length":
        return np.full_like(rest, float(k), dtype=np.float32)
    # strain-normalized spring: 0.5 * k * rest^p * ((l-r)/r)^2
    p = float(args.edge_weight_power)
    return (float(k) * np.power(np.maximum(rest, eps), p) / np.maximum(rest * rest, eps)).astype(np.float32)



def make_floor_point_cloud(x_dyn: Array, floor_height: float, n_env: int, padding: float) -> Tuple[Array, Array]:
    xz = x_dyn[:, [0, 2]]
    lo = xz.min(axis=0) - float(padding)
    hi = xz.max(axis=0) + float(padding)
    side = int(math.ceil(math.sqrt(int(n_env))))
    xs = np.linspace(lo[0], hi[0], side, dtype=np.float32)
    zs = np.linspace(lo[1], hi[1], side, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, zs, indexing="ij"), axis=-1).reshape(-1, 2)[:int(n_env)]
    pts = np.zeros((grid.shape[0], 3), dtype=np.float32)
    pts[:, 0] = grid[:, 0]; pts[:, 1] = float(floor_height); pts[:, 2] = grid[:, 1]
    normals = np.zeros_like(pts); normals[:, 1] = 1.0
    return pts, normals


def choose_force_mask(x: Array, rng: np.random.RandomState, mode: str, fraction: float) -> Array:
    n = int(x.shape[0]); mask = np.zeros(n, dtype=np.float32)
    if n == 0 or mode == "none":
        return mask
    if mode == "all":
        mask[:] = 1.0; return mask
    m = max(1, min(n, int(round(float(fraction) * n))))
    if mode == "top":
        mask[np.argsort(x[:, 1])[-m:]] = 1.0
    elif mode == "bottom":
        mask[np.argsort(x[:, 1])[:m]] = 1.0
    elif mode == "local_patch":
        c = int(rng.randint(0, n)); d2 = np.sum((x - x[c]) ** 2, axis=1); mask[np.argsort(d2)[:m]] = 1.0
    else:
        idx = rng.choice(np.arange(n), size=m, replace=False); mask[idx] = 1.0
    return mask.astype(np.float32)


def force_targets(mode: str, n_objects: int, rng: np.random.RandomState, selected: str) -> List[int]:
    mode = str(mode)
    if mode == "none":
        return []
    if mode.startswith("obj") and mode.endswith("_only"):
        try:
            oid = int(mode[3:-5])
            return [oid] if 0 <= oid < n_objects else []
        except Exception:
            return []
    if mode == "one_of_objects":
        return [int(rng.randint(0, n_objects))]
    if mode == "two_of_objects":
        m = min(2, n_objects)
        return sorted(rng.choice(np.arange(n_objects), size=m, replace=False).astype(int).tolist())
    if mode in ("all", "both"):
        return list(range(n_objects))
    if mode == "selected":
        return sorted(set([v for v in parse_int_list(selected) if 0 <= v < n_objects]))
    return [int(rng.randint(0, n_objects))]


def force_window(T: int, rng: np.random.RandomState, args: argparse.Namespace) -> Tuple[int, int]:
    lo = max(1, int(args.force_duration_min)); hi = max(lo, int(args.force_duration_max))
    length = int(rng.randint(lo, hi + 1)) if hi > lo else lo
    smin = max(0, int(args.force_start_min)); smax = min(max(smin, int(args.force_start_max)), max(0, T - length))
    start = int(rng.randint(smin, smax + 1)) if smax > smin else smin
    return start, min(T, start + length)


def force_direction(oid: int, mode: str, centers: Array, inward_dirs: Array, rng: np.random.RandomState) -> Array:
    if mode == "random_horizontal":
        return random_horizontal_unit(rng)
    if mode == "push_towards_center":
        d = -centers[oid]
    elif mode == "push_away_center":
        d = centers[oid]
    elif mode == "initial_velocity":
        d = inward_dirs[oid]
    elif mode == "opposite_initial_velocity":
        d = -inward_dirs[oid]
    elif mode == "push_towards_next":
        d = centers[(oid + 1) % len(centers)] - centers[oid]
    elif mode == "push_towards_prev":
        d = centers[(oid - 1) % len(centers)] - centers[oid]
    elif mode == "upward":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elif mode == "downward":
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        return random_horizontal_unit(rng)
    d = np.asarray(d, dtype=np.float32); d[1] = 0.0
    return unit(d, random_horizontal_unit(rng))


def build_extra_force(x0: Array, object_id: Array, centers: Array, inward_dirs: Array, rng: np.random.RandomState, args: argparse.Namespace) -> Tuple[Array, Array, Dict[str, object]]:
    T, n = int(args.T), int(x0.shape[0])
    ext = np.zeros((T, n, 3), dtype=np.float32); mask = np.zeros((T, n), dtype=np.float32)
    meta: Dict[str, object] = {"force_active": 0, "force_start": 0, "force_end": 0, "force_magnitude": 0.0,
                               "force_target_objects": np.full((int(args.n_objects),), -1, dtype=np.int64)}
    if str(args.force_mode) == "none" or (float(args.force_prob) < 1.0 and rng.rand() > float(args.force_prob)):
        return ext, mask, meta
    targets = force_targets(str(args.force_mode), int(args.n_objects), rng, str(args.force_target_objects))
    if not targets:
        return ext, mask, meta
    start, end = force_window(T, rng, args)
    mag = float(rng.uniform(float(args.force_mag_min), float(args.force_mag_max)))
    dirs = {}
    for oid in targets:
        dirs[oid] = force_direction(oid, str(args.force_direction), centers, inward_dirs, rng)
        idx = np.where(object_id == oid)[0]
        local_mask = choose_force_mask(x0[idx], rng, str(args.force_mask_mode), float(args.force_mask_fraction))
        if local_mask.sum() <= 0:
            continue
        f_node = mag * dirs[oid][None, :] * local_mask[:, None]
        if bool(args.force_total):
            f_node = f_node / max(float(local_mask.sum()), 1.0)
        ext[start:end, idx] += f_node[None]
        mask[start:end, idx] = np.maximum(mask[start:end, idx], local_mask[None])
    if mask.sum() > 0:
        target_arr = np.full((int(args.n_objects),), -1, dtype=np.int64)
        for i, t in enumerate(sorted(set(targets))[:int(args.n_objects)]):
            target_arr[i] = int(t)
        meta.update({"force_active": 1, "force_start": int(start), "force_end": int(end), "force_magnitude": float(mag), "force_target_objects": target_arr})
        for oid in range(int(args.n_objects)):
            meta[f"force_direction_obj{oid}"] = dirs.get(oid, np.zeros(3, dtype=np.float32)).astype(np.float32)
    return ext.astype(np.float32), mask.astype(np.float32), meta



@dataclass
class ShapeGeom:
    h5_path: str
    shape_index: int
    x_rest: Array
    fps_idx: Array
    norm_scale: float
    norm_center: Array
    meta: Dict[str, object]


@dataclass
class InitState:
    x0: Array
    v0: Array
    object_id: Array
    centers: Array
    inward_dirs: Array
    layout: str
    target_contact_time: float
    relative_speed: float
    yaws: Array


def horizontal_radius(x: Array) -> float:
    return float(np.max(np.linalg.norm(x[:, [0, 2]], axis=1)))


def prepare_initial_state(rest_list: Sequence[Array], rng: np.random.RandomState, args: argparse.Namespace) -> InitState:
    M = int(args.n_objects)
    yawed: List[Array] = []
    yaws = np.zeros(M, dtype=np.float32)
    for oid in range(M):
        x = np.asarray(rest_list[oid], dtype=np.float32).copy()
        yaw = float(rng.uniform(0, 2 * math.pi)) if bool(args.random_yaw) else 0.0
        yaws[oid] = yaw
        x = (x @ rotation_yaw(yaw).T).astype(np.float32)
        if float(args.initial_deformation_std) > 0:
            x += rng.normal(scale=float(args.initial_deformation_std), size=x.shape).astype(np.float32)
        yawed.append(x)
    tau = float(rng.uniform(float(args.contact_time_min), float(args.contact_time_max)))
    speed = float(rng.uniform(float(args.relative_speed_min), float(args.relative_speed_max)))
    base_angle = float(rng.uniform(0, 2 * math.pi))
    layout = str(args.layout).lower()
    dirs = np.zeros((M, 3), dtype=np.float32)
    centers = np.zeros((M, 3), dtype=np.float32)
    if layout == "chain" and M >= 2:
        axis = random_horizontal_unit(rng)
        # Place objects along a line and move all toward the mean center.  For odd M,
        # the middle object starts near center with small velocity.
        offsets = np.linspace(-1.0, 1.0, M)
        max_rad = max(horizontal_radius(x) for x in yawed)
        spacing = 2.0 * max_rad + speed * tau + float(args.chain_extra_spacing)
        for oid, off in enumerate(offsets):
            centers[oid] = axis * float(off) * spacing
            dirs[oid] = unit(-centers[oid], -np.sign(off) * axis if abs(off) > 1e-6 else random_horizontal_unit(rng))
    else:
        for oid in range(M):
            ang = base_angle + 2.0 * math.pi * oid / max(M, 1)
            out = np.array([math.cos(ang), 0.0, math.sin(ang)], dtype=np.float32)
            rad = horizontal_radius(yawed[oid])
            R = rad + speed * tau + float(args.ring_extra_radius)
            centers[oid] = out * R
            dirs[oid] = -out
    xs: List[Array] = []
    vs: List[Array] = []
    oids: List[Array] = []
    for oid in range(M):
        x = yawed[oid] + centers[oid][None, :]
        v = dirs[oid][None, :] * speed
        if layout == "chain" and np.linalg.norm(centers[oid]) < 1e-6:
            v = np.zeros((1, 3), dtype=np.float32)
        xs.append(x.astype(np.float32)); vs.append(np.repeat(v.astype(np.float32), x.shape[0], axis=0)); oids.append(np.full((x.shape[0],), oid, dtype=np.int64))
    x0 = np.concatenate(xs, axis=0); v0 = np.concatenate(vs, axis=0); object_id = np.concatenate(oids, axis=0)
    if str(args.floor_mode) != "none":
        miny = float(x0[:, 1].min())
        lift = float(args.floor_height) + float(args.particle_radius) - miny + float(args.floor_lift)
        x0[:, 1] += lift
        centers[:, 1] += lift
    return InitState(x0=x0.astype(np.float32), v0=v0.astype(np.float32), object_id=object_id, centers=centers.astype(np.float32), inward_dirs=dirs.astype(np.float32), layout=layout, target_contact_time=tau, relative_speed=speed, yaws=yaws)


def build_global_springs(rest_list: Sequence[Array], k_values: Sequence[float], args: argparse.Namespace, rng: np.random.RandomState) -> Tuple[Array, Array, Array, Array]:
    edges_all: List[Array] = []; rest_all: List[Array] = []; coef_k1_all: List[Array] = []; oid_all: List[Array] = []
    offset = 0
    for oid, x_rest in enumerate(rest_list):
        edges, rest = build_spring_graph(x_rest, args, rng)
        if edges.size:
            coeff = spring_coefficients(rest, 1.0, args)
            edges_all.append(edges + offset)
            rest_all.append(rest)
            coef_k1_all.append(coeff)
            oid_all.append(np.full((edges.shape[0],), oid, dtype=np.int64))
        offset += x_rest.shape[0]
    if not edges_all:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(edges_all, axis=0), np.concatenate(rest_all).astype(np.float32), np.concatenate(coef_k1_all).astype(np.float32), np.concatenate(oid_all).astype(np.int64)


def pair_contact_edges(x: Array, idx_a: Array, idx_b: Array, k: int) -> List[Tuple[int, int]]:
    if len(idx_a) == 0 or len(idx_b) == 0 or k <= 0:
        return []
    xa, xb = x[idx_a], x[idx_b]
    edges = set()
    kk_ab = min(k, len(idx_b)); kk_ba = min(k, len(idx_a))
    if _HAS_SCIPY:
        tree_b = cKDTree(xb)
        _, nn = tree_b.query(xa, k=kk_ab)
        nn = np.asarray(nn)
        if nn.ndim == 1:
            nn = nn[:, None]
        for ii in range(len(idx_a)):
            for jj in nn[ii]:
                edges.add((int(idx_a[ii]), int(idx_b[int(jj)])))
        tree_a = cKDTree(xa)
        _, nn2 = tree_a.query(xb, k=kk_ba)
        nn2 = np.asarray(nn2)
        if nn2.ndim == 1:
            nn2 = nn2[:, None]
        for jj in range(len(idx_b)):
            for ii in nn2[jj]:
                edges.add((int(idx_a[int(ii)]), int(idx_b[jj])))
    else:
        d = pairwise_d2(xa, xb)
        nn = np.argsort(d, axis=1)[:, :kk_ab]
        for ii in range(len(idx_a)):
            for jj in nn[ii]:
                edges.add((int(idx_a[ii]), int(idx_b[int(jj)])))
        nn2 = np.argsort(d.T, axis=1)[:, :kk_ba]
        for jj in range(len(idx_b)):
            for ii in nn2[jj]:
                edges.add((int(idx_a[int(ii)]), int(idx_b[jj])))
    return list(edges)


def compute_forces(x: Array, v: Array, t_frame: int, object_id: Array, edges: Array, rest: Array, edge_oid: Array, k_values: Array, ext_force: Array, gravity: Array, args: argparse.Namespace) -> Tuple[Array, Dict[str, float]]:
    n = x.shape[0]
    F = np.zeros_like(x, dtype=np.float32)
    mass = np.ones((n,), dtype=np.float32)
    if str(args.gravity_mode) != "never":
        F += mass[:, None] * gravity[None, :]
    if ext_force is not None:
        F += ext_force.astype(np.float32)
    if edges.size:
        i = edges[:, 0]; j = edges[:, 1]
        dx = x[i] - x[j]
        length = np.linalg.norm(dx, axis=1) + 1e-8
        d = dx / length[:, None]
        stretch = length - rest
        if str(args.spring_energy_mode) == "length":
            k_edge = k_values[edge_oid].astype(np.float32)
        else:
            eps_rest = float(args.spring_rest_eps)
            pwr = float(args.edge_weight_power)
            k_edge = (k_values[edge_oid].astype(np.float32) * np.power(np.maximum(rest, eps_rest), pwr) / np.maximum(rest * rest, eps_rest)).astype(np.float32)
        # per-edge damping ratio scaled by sqrt(k_edge)
        c_edge = float(args.spring_damping_ratio) * 2.0 * np.sqrt(np.maximum(k_edge, 1e-8))
        relv = np.sum((v[i] - v[j]) * d, axis=1)
        f = (-k_edge * stretch - c_edge * relv)[:, None] * d
        np.add.at(F, i, f)
        np.add.at(F, j, -f)
    pair_max_pen = 0.0; pair_contacts = 0
    margin = 2.0 * float(args.particle_radius) * float(args.pair_contact_margin_factor)
    c_pair = float(args.pair_contact_damping_ratio) * 2.0 * math.sqrt(max(float(args.k_pair_contact_stiffness), 1e-8))
    M = int(args.n_objects)
    for a in range(M):
        idx_a = np.where(object_id == a)[0]
        for b in range(a + 1, M):
            idx_b = np.where(object_id == b)[0]
            for ii, jj in pair_contact_edges(x, idx_a, idx_b, int(args.pair_contact_k)):
                dx = x[ii] - x[jj]
                dist = float(np.linalg.norm(dx) + 1e-8)
                delta = margin - dist
                if delta <= 0:
                    continue
                nrm = dx / dist
                relv_n = float(np.dot(v[ii] - v[jj], nrm))
                fmag = float(args.k_pair_contact_stiffness) * delta
                if relv_n < 0:
                    fmag += -c_pair * relv_n
                f = fmag * nrm
                F[ii] += f; F[jj] -= f
                pair_contacts += 1; pair_max_pen = max(pair_max_pen, float(delta))
    floor_max_pen = 0.0; floor_contacts = 0
    if str(args.floor_mode) != "none":
        y_gap = x[:, 1] - float(args.floor_height)
        delta = float(args.particle_radius) * float(args.floor_contact_margin_factor) - y_gap
        active = np.where(delta > 0)[0]
        if active.size:
            c_floor = float(args.floor_contact_damping_ratio) * 2.0 * math.sqrt(max(float(args.k_floor), 1e-8))
            for ii in active:
                fmag = float(args.k_floor) * float(delta[ii])
                if v[ii, 1] < 0:
                    fmag += -c_floor * float(v[ii, 1])
                F[ii, 1] += fmag
            floor_contacts = int(active.size); floor_max_pen = float(np.max(delta[active]))
    if float(args.sim_force_clip) > 0:
        norm = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
        F = F * np.minimum(1.0, float(args.sim_force_clip) / norm)
    return F.astype(np.float32), {"pair_contacts": float(pair_contacts), "pair_max_pen": pair_max_pen, "floor_contacts": float(floor_contacts), "floor_max_pen": floor_max_pen}


def simulate(x0: Array, v0: Array, object_id: Array, edges: Array, rest: Array, edge_oid: Array, k_values: Array, extra_force: Array, gravity: Array, args: argparse.Namespace) -> Tuple[Array, Array, Dict[str, object]]:
    T = int(args.T); dt = float(args.dt); sub = max(1, int(args.substeps)); h = dt / sub
    x = x0.copy().astype(np.float32); v = v0.copy().astype(np.float32)
    pos = np.zeros((T, x.shape[0], 3), dtype=np.float32)
    vel_inst = np.zeros_like(pos)
    pair_max = 0.0; floor_max = 0.0
    first_pair = -1; first_floor = -1
    for t in range(T):
        pos[t] = x; vel_inst[t] = v
        if t == T - 1:
            break
        for _ in range(sub):
            F, d = compute_forces(x, v, t, object_id, edges, rest, edge_oid, k_values, extra_force[t], gravity, args)
            if first_pair < 0 and d["pair_contacts"] > 0:
                first_pair = t
            if first_floor < 0 and d["floor_contacts"] > 0:
                first_floor = t
            pair_max = max(pair_max, float(d["pair_max_pen"]))
            floor_max = max(floor_max, float(d["floor_max_pen"]))
            a = F  # mass=1 per particle
            if float(args.sim_acc_clip) > 0:
                an = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
                a = a * np.minimum(1.0, float(args.sim_acc_clip) / an)
            v = v + h * a
            x = x + h * v
    vel = np.zeros_like(pos)
    if T > 1:
        vel[1:] = (pos[1:] - pos[:-1]) / dt
        vel[0] = vel_inst[0]
    else:
        vel[0] = vel_inst[0]
    rms_strain = 0.0
    if edges.size:
        i, j = edges[:, 0], edges[:, 1]
        eps = (np.linalg.norm(pos[:, i] - pos[:, j], axis=-1) - rest[None, :]) / np.maximum(rest[None, :], 1e-8)
        rms_strain = float(np.max(np.sqrt(np.mean(eps ** 2, axis=1))))
    return pos, vel, {"vel_inst": vel_inst, "first_pair": int(first_pair), "first_floor": int(first_floor), "pair_max": float(pair_max), "floor_max": float(floor_max), "rms_strain_max": float(rms_strain)}



def append_floor(pos_dyn: Array, vel_dyn: Array, x_init_dyn: Array, floor_pts: Optional[Array], floor_normals: Optional[Array]) -> Tuple[Array, Array, Array, Array, Array]:
    T, ndyn, _ = pos_dyn.shape
    if floor_pts is None or floor_pts.size == 0:
        types = np.zeros((ndyn,), dtype=np.int64)
        env_normals = np.zeros((ndyn, 3), dtype=np.float32)
        return pos_dyn, vel_dyn, x_init_dyn, types, env_normals
    nenv = floor_pts.shape[0]
    pos_env = np.repeat(floor_pts[None], T, axis=0)
    vel_env = np.zeros_like(pos_env)
    pos = np.concatenate([pos_dyn, pos_env], axis=1)
    vel = np.concatenate([vel_dyn, vel_env], axis=1)
    x_init = np.concatenate([x_init_dyn, floor_pts], axis=0)
    types = np.concatenate([np.zeros((ndyn,), dtype=np.int64), np.ones((nenv,), dtype=np.int64)], axis=0)
    env_normals = np.zeros((ndyn + nenv, 3), dtype=np.float32)
    env_normals[ndyn:] = floor_normals
    return pos, vel, x_init, types, env_normals


def save_chunk(dst_dir: str, chunk_id: int, episodes: List[Dict[str, object]], args: argparse.Namespace) -> None:
    if not episodes:
        return
    keys = episodes[0].keys()
    out: Dict[str, object] = {}
    # variable length strings as object arrays; numeric arrays stacked
    for k in keys:
        vals = [ep[k] for ep in episodes]
        if k in {"types", "object_id", "env_normals"}:
            out[k] = vals[0]
        elif k in {"source_h5", "source_h5_list"}:
            out[k] = np.asarray(vals, dtype=object)
        else:
            try:
                out[k] = np.stack(vals, axis=0)
            except Exception:
                out[k] = np.asarray(vals, dtype=object)
    out["dt"] = np.float32(args.dt)
    out["particle_radius"] = np.float32(args.particle_radius)
    out["n_objects"] = np.int64(args.n_objects)
    out["num_stored_states"] = np.int64(args.T)
    out["num_transitions"] = np.int64(int(args.T) - 1)
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, f"dataset_multiobj{args.n_objects}_chunk_{chunk_id:04d}.npz")
    np.savez_compressed(path, **out)
    print(f"[save] {path} episodes={len(episodes)}")


def build_dataset(args: argparse.Namespace) -> None:
    if int(args.T) < 49:
        raise ValueError("The true-H48 protocol requires at least 49 stored states; use --T 49 or larger")
    if glob.glob(os.path.join(args.dst_dir, "*.npz")):
        raise FileExistsError("Refusing to overwrite existing NPZ data; choose a new destination for 49-state trajectories")
    rng = np.random.RandomState(int(args.seed))
    h5s = sorted(glob.glob(os.path.join(args.src_h5_dir, "*.h5")))
    if args.max_files and args.max_files > 0:
        h5s = h5s[: int(args.max_files)]
    if len(h5s) < int(args.n_objects):
        raise RuntimeError(f"Need at least {args.n_objects} h5 files, found {len(h5s)}")
    print(f"[load] {len(h5s)} h5 files")
    geoms: List[ShapeGeom] = []
    for si, p in enumerate(tqdm(h5s, desc="geometry")):
        xraw, meta = read_h5_geometry(p)
        xf, idx = farthest_point_sample(xraw, int(args.target_n_obj_each), seed=int(args.seed) + si * 13)
        xn, scale, center = normalize_object(xf, float(args.normalize_diameter))
        geoms.append(ShapeGeom(p, si, xn, idx, scale, center, meta))
    k_tuples = parse_k_tuples(args.k_values or default_k_tuples(int(args.n_objects)), int(args.n_objects))
    print(f"[k] {len(k_tuples)} tuples: {k_tuples[:5]}{'...' if len(k_tuples)>5 else ''}")
    episodes: List[Dict[str, object]] = []
    chunk_id = 0
    group_count = int(args.max_groups if args.max_groups > 0 else args.max_pairs if args.max_pairs > 0 else 100)
    for group_idx in tqdm(range(group_count), desc=f"groups-{args.n_objects}"):
        shape_ids = rng.choice(np.arange(len(geoms)), size=int(args.n_objects), replace=False)
        rest_list = [geoms[int(i)].x_rest for i in shape_ids]
        for rep in range(int(args.repeat_per_group)):
            init_rng = np.random.RandomState(int(args.seed) + 100003 * group_idx + 9176 * rep)
            init = prepare_initial_state(rest_list, init_rng, args)
            ext_base, ext_mask_base, fmeta = build_extra_force(init.x0, init.object_id, init.centers, init.inward_dirs, init_rng, args)
            floor_pts: Optional[Array] = None; floor_normals: Optional[Array] = None
            if str(args.floor_mode) != "none":
                floor_pts, floor_normals = make_floor_point_cloud(init.x0, float(args.floor_height), int(args.target_n_env), float(args.floor_padding))
            gravity = np.array([0.0, float(args.gravity_y), 0.0], dtype=np.float32) if str(args.gravity_mode) != "never" else np.zeros(3, dtype=np.float32)
            for k_idx, kvals in enumerate(k_tuples):
                sim_rng = np.random.RandomState(int(args.seed) + 33 * group_idx + 71 * rep + 1009 * k_idx)
                edges, rest, coef_k1, edge_oid = build_global_springs(rest_list, np.asarray(kvals, dtype=np.float32), args, sim_rng)
                pos_dyn, vel_dyn, diag = simulate(init.x0, init.v0, init.object_id, edges, rest, edge_oid, np.asarray(kvals, dtype=np.float32), ext_base, gravity, args)
                pos, vel, x_init, types, env_normals = append_floor(pos_dyn, vel_dyn, np.concatenate([x for x in [init.x0]], axis=0), floor_pts, floor_normals)
                if floor_pts is not None and floor_pts.size:
                    object_id = np.concatenate([init.object_id, -np.ones((floor_pts.shape[0],), dtype=np.int64)], axis=0)
                else:
                    object_id = init.object_id.copy()
                n_total = pos.shape[1]; n_obj = init.x0.shape[0]
                extra_force = np.zeros((int(args.T), n_total, 3), dtype=np.float32); extra_mask = np.zeros((int(args.T), n_total), dtype=np.float32)
                extra_force[:, :n_obj] = ext_base; extra_mask[:, :n_obj] = ext_mask_base
                gravity_force = np.zeros_like(extra_force); gravity_mask = np.zeros_like(extra_mask)
                if str(args.gravity_mode) != "never":
                    gravity_force[:, :n_obj] = gravity[None, None, :]
                    gravity_mask[:, :n_obj] = 1.0
                total_force = extra_force + gravity_force
                total_mask = ((extra_mask > 0) | (gravity_mask > 0)).astype(np.float32)
                if str(args.ext_force_save_mode) == "total":
                    ext_force, ext_mask = total_force, total_mask
                else:
                    ext_force, ext_mask = extra_force, extra_mask
                ep: Dict[str, object] = {
                    "pos": pos.astype(np.float32), "vel": vel.astype(np.float32), "vel_inst": diag["vel_inst"],
                    "types": types.astype(np.int64), "object_id": object_id.astype(np.int64), "x_init": x_init.astype(np.float32),
                    "k_values": np.asarray(kvals, dtype=np.float32), "k_triplet": np.asarray(kvals, dtype=np.float32), "k": np.asarray(kvals, dtype=np.float32), "E": np.asarray(kvals, dtype=np.float32),
                    "extra_force": extra_force, "extra_mask": extra_mask, "gravity_force": gravity_force, "gravity_mask": gravity_mask,
                    "total_force": total_force, "total_mask": total_mask, "ext_force": ext_force, "ext_mask": ext_mask,
                    "floor_height": np.float32(args.floor_height), "env_normals": env_normals.astype(np.float32),
                    "has_floor": np.int64(str(args.floor_mode) != "none"), "has_gravity": np.int64(str(args.gravity_mode) != "never"), "has_external_force": np.int64(fmeta.get("force_active", 0)),
                    "n_obj": np.int64(n_obj), "n_env": np.int64(n_total - n_obj), "n_obj_each": np.int64(args.target_n_obj_each),
                    "group_index": np.int64(group_idx), "multi_index": np.int64(group_idx), "repeat_index": np.int64(rep), "k_tuple_index": np.int64(k_idx),
                    "shape_indices": np.asarray(shape_ids, dtype=np.int64),
                    "source_h5_list": np.asarray([geoms[int(i)].h5_path for i in shape_ids], dtype=object),
                    "layout": np.asarray(init.layout), "target_contact_time": np.float32(init.target_contact_time), "relative_speed": np.float32(init.relative_speed),
                    "first_obj_obj_contact_frame": np.int64(diag["first_pair"]), "first_floor_contact_frame": np.int64(diag["first_floor"]),
                    "first_any_contact_frame": np.int64(min([v for v in [diag["first_pair"], diag["first_floor"]] if int(v) >= 0], default=-1)),
                    "diag_max_pair_penetration_full": np.float32(diag["pair_max"]), "diag_max_floor_penetration_full": np.float32(diag["floor_max"]), "rms_strain_max": np.float32(diag["rms_strain_max"]),
                    "spring_edges": edges.astype(np.int64), "spring_rest_lengths": rest.astype(np.float32), "spring_edge_coef_k1": coef_k1.astype(np.float32), "spring_edge_object_id": edge_oid.astype(np.int64),
                    "force_active": np.int64(fmeta.get("force_active", 0)), "force_start": np.int64(fmeta.get("force_start", 0)), "force_end": np.int64(fmeta.get("force_end", 0)), "force_magnitude": np.float32(fmeta.get("force_magnitude", 0.0)), "force_target_objects": fmeta.get("force_target_objects", np.full((int(args.n_objects),), -1, dtype=np.int64)),
                }
                episodes.append(ep)
                if len(episodes) >= int(args.episodes_per_npz):
                    save_chunk(args.dst_dir, chunk_id, episodes, args); episodes = []; chunk_id += 1
    if episodes:
        save_chunk(args.dst_dir, chunk_id, episodes, args)
    with open(os.path.join(args.dst_dir, "dataset_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[done] saved to {args.dst_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build N-object spring-mass collision dataset from PhysCtrl h5 geometries.")
    p.add_argument("--src_h5_dir", required=True)
    p.add_argument("--dst_dir", required=True)
    p.add_argument("--n_objects", type=int, choices=[2,3,5], default=3)
    p.add_argument("--max_files", type=int, default=400)
    p.add_argument("--max_groups", type=int, default=400)
    p.add_argument("--max_pairs", type=int, default=0, help="alias for max_groups")
    p.add_argument("--repeat_per_group", type=int, default=1)
    p.add_argument("--repeat_per_pair", type=int, default=0, help="alias for repeat_per_group")
    p.add_argument("--target_n_obj_each", type=int, default=128)
    p.add_argument("--target_n_env", type=int, default=1024)
    p.add_argument("--T", type=int, default=49, help="Stored states including the initial state; true H48 requires at least 49")
    p.add_argument("--dt", type=float, default=0.0416667)
    p.add_argument("--substeps", type=int, default=64)
    p.add_argument("--episodes_per_npz", type=int, default=64)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--k_values", default="")
    p.add_argument("--normalize_diameter", type=float, default=1.0)
    p.add_argument("--particle_radius", type=float, default=0.04)
    p.add_argument("--random_yaw", type=str2bool, default=True)
    p.add_argument("--initial_deformation_std", type=float, default=0.0)
    p.add_argument("--graph_preset", default="solid", choices=["local", "solid", "rigidish", "custom"])
    p.add_argument("--graph_k", type=int, default=18)
    p.add_argument("--add_mst", type=str2bool, default=True)
    p.add_argument("--long_k", type=int, default=8)
    p.add_argument("--long_start_rank", type=int, default=12)
    p.add_argument("--long_end_rank", type=int, default=96)
    p.add_argument("--long_strategy", default="random", choices=["random", "even", "nearest"])
    p.add_argument("--spring_energy_mode", default="strain", choices=["strain", "length"])
    p.add_argument("--edge_weight_power", type=float, default=1.0)
    p.add_argument("--spring_rest_eps", type=float, default=1e-6)
    p.add_argument("--spring_damping_ratio", type=float, default=0.015)
    p.add_argument("--layout", default="ring", choices=["ring", "chain"])
    p.add_argument("--relative_speed_min", type=float, default=1.0)
    p.add_argument("--relative_speed_max", type=float, default=1.8)
    p.add_argument("--contact_time_min", type=float, default=0.25)
    p.add_argument("--contact_time_max", type=float, default=0.45)
    p.add_argument("--ring_extra_radius", type=float, default=0.0)
    p.add_argument("--chain_extra_spacing", type=float, default=0.0)
    p.add_argument("--floor_mode", default="grid", choices=["none", "grid"])
    p.add_argument("--floor_height", type=float, default=0.0)
    p.add_argument("--floor_lift", type=float, default=0.35)
    p.add_argument("--floor_padding", type=float, default=0.8)
    p.add_argument("--gravity_mode", default="always", choices=["never", "always"])
    p.add_argument("--gravity_y", type=float, default=-4.0)
    p.add_argument("--pair_contact_k", type=int, default=4)
    p.add_argument("--k_pair_contact_stiffness", type=float, default=60000.0)
    p.add_argument("--pair_contact_damping_ratio", type=float, default=0.20)
    p.add_argument("--pair_contact_margin_factor", type=float, default=1.0)
    p.add_argument("--k_floor", type=float, default=80000.0)
    p.add_argument("--floor_contact_damping_ratio", type=float, default=0.20)
    p.add_argument("--floor_contact_margin_factor", type=float, default=1.0)
    p.add_argument("--sim_force_clip", type=float, default=0.0)
    p.add_argument("--sim_acc_clip", type=float, default=0.0)
    p.add_argument("--force_mode", default="one_of_objects")
    p.add_argument("--force_prob", type=float, default=0.5)
    p.add_argument("--force_target_objects", default="")
    p.add_argument("--force_direction", default="push_towards_center")
    p.add_argument("--force_mask_mode", default="local_patch")
    p.add_argument("--force_mask_fraction", type=float, default=0.10)
    p.add_argument("--force_mag_min", type=float, default=2.0)
    p.add_argument("--force_mag_max", type=float, default=8.0)
    p.add_argument("--force_start_min", type=int, default=3)
    p.add_argument("--force_start_max", type=int, default=12)
    p.add_argument("--force_duration_min", type=int, default=3)
    p.add_argument("--force_duration_max", type=int, default=8)
    p.add_argument("--force_total", type=str2bool, default=False)
    p.add_argument("--ext_force_save_mode", default="extra", choices=["extra", "total"])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.max_pairs and args.max_pairs > 0:
        args.max_groups = args.max_pairs
    if args.repeat_per_pair and args.repeat_per_pair > 0:
        args.repeat_per_group = args.repeat_per_pair
    if not args.k_values:
        args.k_values = default_k_tuples(int(args.n_objects))
    build_dataset(args)


if __name__ == "__main__":
    main()
