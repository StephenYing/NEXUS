from __future__ import annotations

import glob
import hashlib
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from numpy.lib import format as npformat
import torch
from torch.utils.data import Dataset


def identity_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("batch_size must be 1 for variable graph samples")
    return batch[0]


INFERRED_GRAPH_DEFAULTS: Dict[str, Any] = {
    "k_obj": 18,
    "struct_graph_preset": "solid",
    "struct_add_mst": True,
    "struct_long_k": 8,
    "struct_long_start_rank": 12,
    "struct_long_end_rank": 96,
    "struct_long_strategy": "even",
    "struct_seed_base": 2026,
    "struct_seed_protocol": "visible_hash_v1",
    "edge_coef_mode": "inv_rest",
    "edge_coef_clip_min": 0.0,
    "edge_coef_clip_max": 100.0,
    "struct_rest_eps": 1e-6,
}


def _config_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def resolve_inferred_graph_config(config: Any = None) -> Dict[str, Any]:
    """Resolve the complete graph protocol shared by train/eval/export."""
    if config is None:
        source: Mapping[str, Any] = {}
    elif isinstance(config, Mapping):
        source = config
    else:
        source = vars(config)
    values = {key: source.get(key, default) for key, default in INFERRED_GRAPH_DEFAULTS.items()}
    return {
        "k_obj": int(values["k_obj"]),
        "struct_graph_preset": str(values["struct_graph_preset"]),
        "struct_add_mst": _config_bool(values["struct_add_mst"]),
        "struct_long_k": int(values["struct_long_k"]),
        "struct_long_start_rank": int(values["struct_long_start_rank"]),
        "struct_long_end_rank": int(values["struct_long_end_rank"]),
        "struct_long_strategy": str(values["struct_long_strategy"]),
        "struct_seed_base": int(values["struct_seed_base"]),
        "struct_seed_protocol": str(values["struct_seed_protocol"]),
        "edge_coef_mode": str(values["edge_coef_mode"]),
        "edge_coef_clip_min": float(values["edge_coef_clip_min"]),
        "edge_coef_clip_max": float(values["edge_coef_clip_max"]),
        "struct_rest_eps": float(values["struct_rest_eps"]),
    }


def resolve_checkpoint_inferred_graph_config(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Load a complete graph protocol without silently guessing missing fields."""
    source = checkpoint.get("inferred_graph_config")
    source_name = "inferred_graph_config"
    if source is None:
        source = checkpoint.get("args")
        source_name = "args"
    if not isinstance(source, Mapping):
        raise ValueError("Checkpoint does not contain an inferred graph protocol")
    missing = sorted(set(INFERRED_GRAPH_DEFAULTS) - {"struct_seed_protocol"} - set(source))
    if missing:
        raise ValueError(f"Checkpoint {source_name} is missing graph fields: {missing}")
    source = dict(source)
    if "struct_seed_protocol" not in source:
        # Versioned release checkpoints used visible hashes; archived training
        # predates that release and used file/episode indices for long edges.
        source["struct_seed_protocol"] = "visible_hash_v1" if checkpoint.get("knn_edge_convention") else "legacy_index_v1"
    return resolve_inferred_graph_config(source)


def discover_npz_files(spec: str) -> List[str]:
    if os.path.isdir(spec):
        files = sorted(glob.glob(os.path.join(spec, "*.npz")))
    elif any(c in spec for c in "*?[]"):
        files = sorted(glob.glob(spec))
    else:
        files = [spec]
    files = [f for f in files if f.endswith(".npz") and ".wcache_" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No .npz files found for {spec!r}")
    return files


def _npz_key_shape(path: str, key: str) -> Tuple[int, ...]:
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(f"{key}.npy", "r") as f:
            ver = npformat.read_magic(f)
            if ver == (1, 0):
                shape, _, _ = npformat.read_array_header_1_0(f)
            elif ver in ((2, 0), (3, 0)):
                shape, _, _ = npformat.read_array_header_2_0(f)
            else:
                raise ValueError(f"Unsupported npy version {ver} in {path}:{key}")
    return tuple(int(x) for x in shape)


def _first_key(raw: Any, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in raw:
            return k
    return None


def _as_traj(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[-1] == 3:
        x = x[None]
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"{name} must be [E,T,N,3] or [T,N,3], got {x.shape}")
    return x


def _scalar_float(x: Any, default: float) -> float:
    if x is None:
        return float(default)
    a = np.asarray(x)
    if a.size == 0:
        return float(default)
    return float(a.reshape(-1)[0])


def _scalar_int(x: Any, default: int) -> int:
    if x is None:
        return int(default)
    a = np.asarray(x)
    if a.size == 0:
        return int(default)
    return int(a.reshape(-1)[0])


def _episode_value(arr: Optional[np.ndarray], ep: int, n_ep: int, default: Any) -> Any:
    if arr is None:
        return default
    a = np.asarray(arr)
    if a.shape == ():
        return a.item()
    if a.ndim >= 1 and a.shape[0] == n_ep:
        return a[ep]
    return a


def _frame_value(arr: Optional[np.ndarray], ep: int, t: int, n_ep: int, n_t: int, default: Any) -> Any:
    if arr is None:
        return default
    a = np.asarray(arr)
    if a.shape == ():
        return a.item()
    if a.ndim >= 2 and a.shape[0] == n_ep and a.shape[1] == n_t:
        return a[ep, t]
    if a.ndim >= 1 and a.shape[0] == n_ep:
        return a[ep]
    return a


def _torch_f32(x: Any) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32).copy())


def _torch_i64(x: Any) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.int64).copy())


def _fix_particle_array(a: Any, n_total: int, n_obj: int, tail_fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((n_total,), dtype=np.float32)
    if arr.ndim == 0:
        return np.full((n_total,), float(arr), dtype=np.float32)
    if arr.shape[0] == n_total:
        return arr.astype(np.float32)
    out = np.full((n_total,) + arr.shape[1:], tail_fill, dtype=np.float32)
    m = min(n_total, arr.shape[0])
    out[:m] = arr[:m]
    return out


def _fix_force_array(a: Any, n_total: int, n_obj: int) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((n_total, 3), dtype=np.float32)
    if arr.ndim == 1 and arr.size == 3:
        out = np.zeros((n_total, 3), dtype=np.float32)
        out[:n_obj] = arr.reshape(1, 3)
        return out
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return _fix_particle_array(arr, n_total, n_obj, tail_fill=0.0).reshape(n_total, 3)
    out = np.zeros((n_total, 3), dtype=np.float32)
    flat = arr.reshape(-1, 3) if arr.size % 3 == 0 else np.zeros((0, 3), dtype=np.float32)
    m = min(n_total, flat.shape[0])
    out[:m] = flat[:m]
    return out


def _fix_mask_array(a: Any, n_total: int, n_obj: int) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((n_total,), dtype=np.float32)
    return _fix_particle_array(arr.reshape(-1), n_total, n_obj, tail_fill=0.0).reshape(n_total)


def _hash_to_unit(s: str) -> float:
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return int(h, 16) / float(16 ** 12 - 1)


def _stable_episode_graph_seed(
    x_init: np.ndarray,
    types: np.ndarray,
    object_id: np.ndarray,
    k_values: np.ndarray,
    seed_base: int,
) -> int:
    """Return a stable seed from the visible episode inputs, independent of file order."""
    ty = np.asarray(types, dtype=np.int64).reshape(-1)
    oid = np.asarray(object_id, dtype=np.int64).reshape(-1)
    x = np.asarray(x_init, dtype=np.float32).reshape(-1, 3)
    if x.shape[0] != ty.size or ty.size != oid.size:
        raise ValueError("x_init/types/object_id sizes do not match for graph seeding")
    dynamic = (ty == 0) & (oid >= 0)
    h = hashlib.sha1()
    h.update(b"multiobj-inferred-graph-seed-v1")
    for arr, dtype in (
        (np.asarray([int(seed_base)]), "<i8"),
        (x[dynamic], "<f4"),
        (oid[dynamic], "<i8"),
        (np.asarray(k_values).reshape(-1), "<f4"),
    ):
        value = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
        h.update(np.asarray(value.shape, dtype="<i8").tobytes())
        h.update(value.tobytes())
    return int.from_bytes(h.digest()[:4], byteorder="little", signed=False)

# Inferred structural graph from x_init and object_id only.


def _pairwise_d2_np(x: np.ndarray) -> np.ndarray:
    xx = np.sum(x * x, axis=1, keepdims=True)
    return np.maximum(xx + xx.T - 2.0 * (x @ x.T), 0.0)


def _canonical_edge(a: int, b: int) -> Tuple[int, int]:
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def _symmetric_knn_edges_np(x: np.ndarray, k: int) -> np.ndarray:
    n = int(x.shape[0])
    if n <= 1 or k <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    k = min(int(k), n - 1)
    d2 = _pairwise_d2_np(x)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    edges = set()
    for i in range(n):
        for j in nn[i]:
            a, b = _canonical_edge(i, int(j))
            if a != b:
                edges.add((a, b))
    return np.asarray(sorted(edges), dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)


def _prim_mst_edges_np(x: np.ndarray) -> np.ndarray:
    n = int(x.shape[0])
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    d2 = _pairwise_d2_np(x)
    in_tree = np.zeros(n, dtype=bool); in_tree[0] = True
    min_d2 = d2[0].copy(); parent = np.zeros(n, dtype=np.int64)
    edges: List[Tuple[int, int]] = []
    for _ in range(n - 1):
        masked = np.where(in_tree, np.inf, min_d2)
        j = int(np.argmin(masked))
        if not np.isfinite(masked[j]):
            break
        i = int(parent[j])
        a, b = _canonical_edge(i, j)
        if a != b:
            edges.append((a, b))
        in_tree[j] = True
        better = d2[j] < min_d2
        parent[better] = j
        min_d2[better] = d2[j, better]
    return np.asarray(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)


def _midrange_edges_np(x: np.ndarray, long_k: int, start_rank: int, end_rank: int, strategy: str = "even", seed: int = 0) -> np.ndarray:
    n = int(x.shape[0])
    if n <= 1 or long_k <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    d2 = _pairwise_d2_np(x)
    np.fill_diagonal(d2, np.inf)
    order = np.argsort(d2, axis=1)
    start = max(1, int(start_rank)); end = min(max(start + 1, int(end_rank)), n)
    long_k = min(int(long_k), max(0, end - start))
    if long_k <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    rng = np.random.RandomState(int(seed)); edges = set(); strategy = str(strategy).lower()
    for i in range(n):
        pool = order[i, start:end]
        pool = pool[np.isfinite(d2[i, pool])]
        if pool.size == 0:
            continue
        if strategy == "random":
            take = rng.choice(pool, size=min(long_k, pool.size), replace=False)
        elif strategy == "nearest":
            take = pool[:long_k]
        else:
            if pool.size <= long_k:
                take = pool
            else:
                idx = np.linspace(0, pool.size - 1, num=long_k, dtype=int)
                take = pool[idx]
        for j in take:
            a, b = _canonical_edge(i, int(j))
            if a != b:
                edges.add((a, b))
    return np.asarray(sorted(edges), dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)


def _union_edges_np(*arrays: np.ndarray) -> np.ndarray:
    edges = set()
    for a in arrays:
        if a is None or np.asarray(a).size == 0:
            continue
        for i, j in np.asarray(a, dtype=np.int64).reshape(-1, 2):
            u, v = _canonical_edge(int(i), int(j))
            if u != v:
                edges.add((u, v))
    return np.asarray(sorted(edges), dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)


def build_inferred_struct_graph(
    x_init: torch.Tensor,
    types: torch.Tensor,
    object_id: torch.Tensor,
    k_obj: int = 18,
    *,
    graph_preset: str = "solid",
    add_mst: bool = True,
    long_k: int = 8,
    long_start_rank: int = 12,
    long_end_rank: int = 96,
    long_strategy: str = "even",
    long_seed: int = 0,
    edge_coef_mode: str = "inv_rest",
    edge_coef_clip_min: float = 0.0,
    edge_coef_clip_max: float = 100.0,
    rest_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x_init.detach().float().cpu()
    ty = types.detach().long().cpu()
    oid = object_id.detach().long().cpu()
    dynamic = torch.where((ty == 0) & (oid >= 0))[0]
    if dynamic.numel() <= 1:
        empty_e = torch.empty((2, 0), dtype=torch.long); empty_f = torch.empty((0,), dtype=torch.float32); empty_o = torch.empty((0,), dtype=torch.long)
        return empty_e, empty_f, empty_f, empty_o
    preset = str(graph_preset or "solid").lower()
    if preset in ("solid", "generator_like", "solid_inferred"):
        add_mst = True
        if int(long_k) <= 0:
            long_k = 8
        if int(k_obj) <= 0:
            k_obj = 18
    elif preset in ("mst", "local_mst"):
        add_mst = True; long_k = 0
    elif preset in ("local", "knn", "pure_knn"):
        add_mst = False; long_k = 0
    elif preset in ("custom", "manual"):
        pass
    else:
        raise ValueError(f"Unknown graph_preset={graph_preset!r}")
    all_edges: List[np.ndarray] = []; all_oid: List[np.ndarray] = []
    unique_oids = sorted([int(o) for o in torch.unique(oid[dynamic]).tolist() if int(o) >= 0])
    for o in unique_oids:
        idx = torch.where((ty == 0) & (oid == o))[0]
        if idx.numel() <= 1:
            continue
        xo = x[idx].numpy().astype(np.float32, copy=False)
        e_knn = _symmetric_knn_edges_np(xo, int(k_obj))
        e_mst = _prim_mst_edges_np(xo) if bool(add_mst) else np.zeros((0, 2), dtype=np.int64)
        e_long = _midrange_edges_np(xo, int(long_k), int(long_start_rank), int(long_end_rank), strategy=str(long_strategy), seed=int(long_seed) + 1009 * int(o)) if int(long_k) > 0 else np.zeros((0, 2), dtype=np.int64)
        e_local = _union_edges_np(e_knn, e_mst, e_long)
        if e_local.size == 0:
            continue
        mapped = idx[torch.as_tensor(e_local.reshape(-1), dtype=torch.long)].view(-1, 2).numpy().astype(np.int64)
        all_edges.append(mapped); all_oid.append(np.full((mapped.shape[0],), int(o), dtype=np.int64))
    if not all_edges:
        empty_e = torch.empty((2, 0), dtype=torch.long); empty_f = torch.empty((0,), dtype=torch.float32); empty_o = torch.empty((0,), dtype=torch.long)
        return empty_e, empty_f, empty_f, empty_o
    e = np.concatenate(all_edges, axis=0); eo = np.concatenate(all_oid, axis=0)
    pairs = {}
    for (a, b), o in zip(e.tolist(), eo.tolist()):
        u, v = _canonical_edge(a, b); pairs[(u, v)] = int(o)
    keys = sorted(pairs.keys())
    e = np.asarray(keys, dtype=np.int64); eo = np.asarray([pairs[k] for k in keys], dtype=np.int64)
    gi = torch.as_tensor(e[:, 0], dtype=torch.long); gj = torch.as_tensor(e[:, 1], dtype=torch.long)
    rest = torch.norm(x[gi] - x[gj], dim=-1).float().clamp_min(float(rest_eps))
    mode = str(edge_coef_mode or "inv_rest").lower()
    if mode in ("none", "one", "ones"):
        coef = torch.ones_like(rest)
    elif mode in ("inv_rest", "strain", "one_over_rest"):
        coef = 1.0 / rest
    else:
        raise ValueError(f"Unknown edge_coef_mode={edge_coef_mode!r}")
    if edge_coef_clip_min and float(edge_coef_clip_min) > 0:
        coef = coef.clamp_min(float(edge_coef_clip_min))
    if edge_coef_clip_max and float(edge_coef_clip_max) > 0:
        coef = coef.clamp_max(float(edge_coef_clip_max))
    return torch.stack([gi, gj], dim=0), rest, coef.float(), torch.as_tensor(eo, dtype=torch.long)


@dataclass
class NPZFile:
    path: str
    keys: Tuple[str, ...]
    pos_key: str
    vel_key: Optional[str]
    n_ep: int
    n_t: int
    n_total: int
    n_obj: int
    n_env: int
    n_objects: int
    types: np.ndarray
    object_id: np.ndarray
    dt: float
    particle_radius: float
    group_index: Optional[np.ndarray]
    repeat_index: Optional[np.ndarray]
    k_tuple_index: Optional[np.ndarray]
    first_obj_obj_contact_frame: Optional[np.ndarray]
    first_floor_contact_frame: Optional[np.ndarray]
    first_any_contact_frame: Optional[np.ndarray]


def _load_file(path: str, dt_override: float = 0.0, particle_radius_override: float = 0.0, n_objects_override: int = 0) -> NPZFile:
    with np.load(path, allow_pickle=True) as raw:
        keys = tuple(raw.files)
        pos_key = _first_key(raw, ["pos", "x", "positions"])
        if pos_key is None:
            raise KeyError(f"{path} has no pos/x/positions key")
        vel_key = _first_key(raw, ["vel", "v", "velocity"])
        pos_shape = _npz_key_shape(path, pos_key)
        if len(pos_shape) == 4 and pos_shape[-1] == 3:
            n_ep, n_t, n_total = int(pos_shape[0]), int(pos_shape[1]), int(pos_shape[2])
        elif len(pos_shape) == 3 and pos_shape[-1] == 3:
            n_ep, n_t, n_total = 1, int(pos_shape[0]), int(pos_shape[1])
        else:
            raise ValueError(f"{path}:{pos_key} bad shape {pos_shape}")
        dt = float(dt_override) if dt_override and dt_override > 0 else _scalar_float(raw["dt"] if "dt" in raw else None, 1.0)
        pr = float(particle_radius_override) if particle_radius_override and particle_radius_override > 0 else _scalar_float(raw["particle_radius"] if "particle_radius" in raw else None, 0.04)
        types = np.asarray(raw["types"], dtype=np.int64).reshape(-1) if "types" in raw else np.zeros((n_total,), dtype=np.int64)
        if types.size != n_total:
            fixed = np.ones((n_total,), dtype=np.int64); fixed[:min(types.size, n_total)] = types[:min(types.size, n_total)]; types = fixed
        if "object_id" in raw:
            object_id = np.asarray(raw["object_id"], dtype=np.int64).reshape(-1)
        else:
            n_obj_guess = _scalar_int(raw["n_obj"] if "n_obj" in raw else None, int((types == 0).sum()))
            n_objects_tmp = int(n_objects_override or _scalar_int(raw["n_objects"] if "n_objects" in raw else None, 1))
            n_each = max(1, n_obj_guess // max(n_objects_tmp, 1))
            object_id = -np.ones((n_total,), dtype=np.int64)
            for o in range(n_objects_tmp):
                object_id[o*n_each:min((o+1)*n_each, n_obj_guess)] = o
        if object_id.size != n_total:
            fixed = -np.ones((n_total,), dtype=np.int64); fixed[:min(object_id.size, n_total)] = object_id[:min(object_id.size, n_total)]; object_id = fixed
        n_objects = int(n_objects_override or _scalar_int(raw["n_objects"] if "n_objects" in raw else None, int(max(1, object_id[object_id >= 0].max() + 1 if np.any(object_id >= 0) else 1))))
        n_obj = _scalar_int(raw["n_obj"] if "n_obj" in raw else None, int((types == 0).sum()))
        n_env = _scalar_int(raw["n_env"] if "n_env" in raw else None, int((types != 0).sum()))
        def small_arr(name: str, max_bytes: int = 16 * 1024 * 1024) -> Optional[np.ndarray]:
            if name not in raw:
                return None
            try:
                shp = _npz_key_shape(path, name); num = int(np.prod(shp)) if shp else 1
                if max(1, num) * 8 > max_bytes:
                    return None
            except Exception:
                pass
            return np.asarray(raw[name])
        group = small_arr("group_index")
        if group is None:
            group = small_arr("multi_index")
        if group is None:
            group = small_arr("pair_index")
        if group is None:
            group = small_arr("triple_index")
        return NPZFile(
            path=path, keys=keys, pos_key=pos_key, vel_key=vel_key, n_ep=n_ep, n_t=n_t, n_total=n_total,
            n_obj=n_obj, n_env=n_env, n_objects=n_objects, types=types, object_id=object_id, dt=dt, particle_radius=pr,
            group_index=group, repeat_index=small_arr("repeat_index"), k_tuple_index=(small_arr("k_tuple_index") if small_arr("k_tuple_index") is not None else (small_arr("k_triplet_index") if small_arr("k_triplet_index") is not None else small_arr("k_pair_index"))),
            first_obj_obj_contact_frame=small_arr("first_obj_obj_contact_frame"), first_floor_contact_frame=small_arr("first_floor_contact_frame"), first_any_contact_frame=small_arr("first_any_contact_frame"),
        )


class MultiObjectDataset(Dataset):
    def __init__(
        self,
        data: str,
        mode: str = "train",
        split: float = 0.9,
        split_by: str = "group_index",
        dt: float = 0.0,
        particle_radius: float = 0.0,
        n_objects: int = 0,
        min_k: float = 0.0,
        max_k: float = 0.0,
        max_windows: int = 0,
        max_files: int = 0,
        load_workers: int = 0,
        worker_cache_size: int = 1,
        rollout_steps: int = 1,
        clip_rollout_to_available: bool = False,
        min_stored_states: int = 0,
        oversample_contact: int = 1,
        contact_window_radius: int = 6,
        contact_oversample_mode: str = "metadata",
        k_obj: int = 18,
        struct_graph_preset: str = "solid",
        struct_add_mst: bool = True,
        struct_long_k: int = 8,
        struct_long_start_rank: int = 12,
        struct_long_end_rank: int = 96,
        struct_long_strategy: str = "even",
        struct_seed_base: int = 2026,
        struct_seed_protocol: str = "visible_hash_v1",
        edge_coef_mode: str = "inv_rest",
        edge_coef_clip_min: float = 0.0,
        edge_coef_clip_max: float = 100.0,
        struct_rest_eps: float = 1e-6,
    ):
        self.data_spec = data; self.mode = str(mode); self.split = float(split); self.split_by = str(split_by)
        self.n_objects_override = int(n_objects)
        self.requested_rollout_steps = max(1, int(rollout_steps))
        self.rollout_steps = self.requested_rollout_steps
        self.clip_rollout_to_available = bool(clip_rollout_to_available)
        self.min_stored_states = max(0, int(min_stored_states))
        self.min_k = float(min_k); self.max_k = float(max_k)
        self.load_workers = int(load_workers); self.worker_cache_size = max(1, int(worker_cache_size))
        self.oversample_contact = int(oversample_contact); self.contact_window_radius = int(contact_window_radius); self.contact_oversample_mode = str(contact_oversample_mode or "metadata").lower()
        self.k_obj = int(k_obj); self.struct_graph_preset = str(struct_graph_preset or "solid"); self.struct_add_mst = bool(struct_add_mst)
        self.struct_long_k = int(struct_long_k); self.struct_long_start_rank = int(struct_long_start_rank); self.struct_long_end_rank = int(struct_long_end_rank)
        self.struct_long_strategy = str(struct_long_strategy or "even"); self.struct_seed_base = int(struct_seed_base)
        self.struct_seed_protocol = str(struct_seed_protocol)
        if self.struct_seed_protocol not in {"visible_hash_v1", "legacy_index_v1"}:
            raise ValueError(f"Unknown structural seed protocol: {self.struct_seed_protocol}")
        self.edge_coef_mode = str(edge_coef_mode or "inv_rest"); self.edge_coef_clip_min = float(edge_coef_clip_min); self.edge_coef_clip_max = float(edge_coef_clip_max); self.struct_rest_eps = float(struct_rest_eps)
        files = discover_npz_files(data)
        if max_files and max_files > 0:
            files = files[:int(max_files)]
        if self.load_workers > 1 and len(files) > 1:
            with ThreadPoolExecutor(max_workers=min(self.load_workers, len(files))) as ex:
                self.files = list(ex.map(lambda f: _load_file(f, dt_override=dt, particle_radius_override=particle_radius, n_objects_override=self.n_objects_override), files))
        else:
            self.files = [_load_file(f, dt_override=dt, particle_radius_override=particle_radius, n_objects_override=self.n_objects_override) for f in files]
        for info in self.files:
            if int(info.n_t) < self.min_stored_states:
                raise ValueError(
                    f"{info.path} contains {info.n_t} stored states; this protocol requires "
                    f"at least {self.min_stored_states}. True H48 needs 49 states including the initial state."
                )
            if not self.clip_rollout_to_available and int(info.n_t) <= self.requested_rollout_steps:
                raise ValueError(
                    f"Requested {self.requested_rollout_steps} transitions, but {info.path} "
                    f"contains only {info.n_t} stored states; at least {self.requested_rollout_steps + 1} are required"
                )
        if self.clip_rollout_to_available and self.files:
            max_common_rollout = min(int(info.n_t) - 1 for info in self.files)
            if max_common_rollout < 1:
                raise ValueError("Evaluation requires at least two states per trajectory")
            self.rollout_steps = min(self.requested_rollout_steps, max_common_rollout)
        self.n_objects = int(self.n_objects_override or (self.files[0].n_objects if self.files else 1))
        self._npz_cache: Dict[int, Dict[str, Any]] = {}; self._npz_cache_order: List[int] = []; self._graph_cache: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.windows = np.zeros((0, 3), dtype=np.int32); self.window_weights = np.zeros((0,), dtype=np.float32); self.window_is_contact = np.zeros((0,), dtype=np.uint8)
        self._build_windows(max_windows=max_windows)

    def _get_slot(self, fi: int) -> Dict[str, Any]:
        slot = self._npz_cache.get(fi)
        if slot is None:
            raw = np.load(self.files[fi].path, allow_pickle=True, mmap_mode="r")
            slot = {"raw": raw, "arr": {}}
            self._npz_cache[fi] = slot
        if fi in self._npz_cache_order:
            self._npz_cache_order.remove(fi)
        self._npz_cache_order.append(fi)
        while len(self._npz_cache_order) > self.worker_cache_size:
            old = self._npz_cache_order.pop(0); old_slot = self._npz_cache.pop(old, None)
            if old_slot is not None:
                try: old_slot["raw"].close()
                except Exception: pass
        return slot

    def _get_array(self, fi: int, key: Optional[str]) -> Optional[np.ndarray]:
        if key is None:
            return None
        info = self.files[fi]
        if key not in info.keys:
            return None
        slot = self._get_slot(fi); arr_cache = slot["arr"]
        if key not in arr_cache:
            arr_cache[key] = np.asarray(slot["raw"][key])
        return arr_cache[key]

    def _get_pos_vel(self, fi: int) -> Tuple[np.ndarray, np.ndarray]:
        info = self.files[fi]
        pos = self._get_array(fi, info.pos_key)
        if pos is None: raise RuntimeError(f"Missing pos in {info.path}")
        pos = _as_traj(pos, info.pos_key)
        vel = self._get_array(fi, info.vel_key)
        if vel is not None:
            return pos, _as_traj(vel, info.vel_key or "vel")
        slot = self._get_slot(fi)
        if "_vel_fd" not in slot["arr"]:
            dt = max(float(info.dt), 1e-8); v = np.zeros_like(pos, dtype=np.float32); v[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / dt; v[:, 0] = v[:, 1] if v.shape[1] > 1 else 0.0; slot["arr"]["_vel_fd"] = v
            print(f"[warn] {os.path.basename(info.path)} missing vel; using finite differences")
        return pos, slot["arr"]["_vel_fd"]

    def _k_values_for_episode(self, fi: int, info: NPZFile, ep: int) -> np.ndarray:
        arr = None
        for key in ("k_values", "k_tuple", "k_triplet", "k_pair", "k", "E"):
            arr = self._get_array(fi, key)
            if arr is not None:
                break
        if arr is None:
            vals = []
            for o in range(self.n_objects):
                ko = self._get_array(fi, f"k{o+1}")
                if ko is None:
                    ko = self._get_array(fi, f"k{o}")
                if ko is None: break
                vals.append(_scalar_float(_episode_value(ko, ep, info.n_ep, 100.0), 100.0))
            if vals:
                a = np.asarray(vals, dtype=np.float32)
            else:
                a = np.full((self.n_objects,), 100.0, dtype=np.float32)
        else:
            val = _episode_value(arr, ep, info.n_ep, np.full((self.n_objects,), 100.0, dtype=np.float32))
            a = np.asarray(val, dtype=np.float32).reshape(-1)
        if a.size >= self.n_objects:
            return a[:self.n_objects].astype(np.float32)
        if a.size == 1:
            return np.full((self.n_objects,), float(a[0]), dtype=np.float32)
        out = np.full((self.n_objects,), float(np.mean(a)) if a.size else 100.0, dtype=np.float32)
        if a.size:
            out[:a.size] = a
        return out.astype(np.float32)

    def _group_key(self, fi: int, info: NPZFile, ep: int) -> str:
        mode = self.split_by.lower()
        if mode == "episode":
            return f"ep:{fi}:{ep}"
        if info.group_index is not None:
            return f"group:{_scalar_int(_episode_value(info.group_index, ep, info.n_ep, ep), ep)}"
        return f"ep:{fi}:{ep}"

    def _is_train_group(self, key: str) -> bool:
        return _hash_to_unit(key) < self.split

    def _contact_frames_for_ep(self, info: NPZFile, ep: int) -> List[int]:
        frames: List[int] = []
        for arr in (info.first_any_contact_frame, info.first_obj_obj_contact_frame, info.first_floor_contact_frame):
            if arr is None: continue
            v = _scalar_int(_episode_value(arr, ep, info.n_ep, -1), -1)
            if v >= 0: frames.append(v)
        return frames

    def _near_contact_metadata(self, info: NPZFile, ep: int, t: int, horizon: int) -> bool:
        if self.contact_oversample_mode in ("none", "off", "0", "false"):
            return False
        rad = max(0, int(self.contact_window_radius))
        for fr in self._contact_frames_for_ep(info, ep):
            if (t <= fr + rad) and (t + horizon >= fr - rad): return True
        if self.contact_oversample_mode in ("time", "prior"):
            center = 0.35 * float(info.n_t); width = max(float(horizon), 0.2 * float(info.n_t)); return abs(float(t) - center) <= width
        return False

    def _build_windows(self, max_windows: int = 0) -> None:
        wins: List[Tuple[int, int, int]] = []; weights: List[float] = []; flags: List[int] = []
        horizon = max(1, int(self.rollout_steps))
        for fi, info in enumerate(self.files):
            for ep in range(info.n_ep):
                kv = self._k_values_for_episode(fi, info, ep)
                if self.min_k > 0.0 and float(np.max(kv)) < self.min_k: continue
                if self.max_k > 0.0 and float(np.min(kv)) > self.max_k: continue
                key = self._group_key(fi, info, ep); train = self._is_train_group(key)
                if self.mode == "train" and not train: continue
                if self.mode in ("valid", "val") and train: continue
                for t in range(max(0, info.n_t - horizon)):
                    near = self._near_contact_metadata(info, ep, t, horizon) if self.oversample_contact > 1 else False
                    wins.append((fi, ep, t)); weights.append(float(self.oversample_contact) if near else 1.0); flags.append(1 if near else 0)
                    if max_windows and max_windows > 0 and len(wins) >= max_windows:
                        self.windows = np.asarray(wins, dtype=np.int32); self.window_weights = np.asarray(weights, dtype=np.float32); self.window_is_contact = np.asarray(flags, dtype=np.uint8); return
        if not wins: raise RuntimeError("No windows after split/filter")
        self.windows = np.asarray(wins, dtype=np.int32); self.window_weights = np.asarray(weights, dtype=np.float32); self.window_is_contact = np.asarray(flags, dtype=np.uint8)

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def sampler_weights(self) -> np.ndarray:
        return self.window_weights.astype(np.float32, copy=False) if self.window_weights.size else np.ones((len(self),), dtype=np.float32)

    def _struct_graph(self, fi: int, info: NPZFile, ep: int, x0_np: np.ndarray, k_values: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            fi, ep, self.struct_graph_preset, self.k_obj, self.struct_add_mst,
            self.struct_long_k, self.struct_long_start_rank, self.struct_long_end_rank,
            self.struct_long_strategy, self.struct_seed_base, self.edge_coef_mode,
            self.edge_coef_clip_min, self.edge_coef_clip_max, self.struct_rest_eps,
            self.n_objects,
        )
        if key not in self._graph_cache:
            if self.struct_seed_protocol == "legacy_index_v1":
                seed = int(self.struct_seed_base) + 100003 * fi + 9176 * ep
            else:
                seed = _stable_episode_graph_seed(
                    x0_np, info.types, info.object_id, k_values, self.struct_seed_base
                )
            self._graph_cache[key] = build_inferred_struct_graph(
                torch.from_numpy(np.asarray(x0_np, dtype=np.float32)), torch.from_numpy(np.asarray(info.types, dtype=np.int64)), torch.from_numpy(np.asarray(info.object_id, dtype=np.int64)), self.k_obj,
                graph_preset=self.struct_graph_preset, add_mst=self.struct_add_mst, long_k=self.struct_long_k, long_start_rank=self.struct_long_start_rank, long_end_rank=self.struct_long_end_rank,
                long_strategy=self.struct_long_strategy, long_seed=seed, edge_coef_mode=self.edge_coef_mode, edge_coef_clip_min=self.edge_coef_clip_min, edge_coef_clip_max=self.edge_coef_clip_max, rest_eps=self.struct_rest_eps)
        return self._graph_cache[key]

    def _env_normals(self, fi: int, info: NPZFile, ep: int) -> np.ndarray:
        arr = self._get_array(fi, "env_normals")
        if arr is None:
            out = np.zeros((info.n_total, 3), dtype=np.float32); out[info.n_obj:, 1] = 1.0; return out
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 3 and a.shape[0] == info.n_ep:
            return _fix_particle_array(a[ep], info.n_total, info.n_obj, tail_fill=0.0).reshape(info.n_total, 3)
        if a.ndim == 2:
            return _fix_particle_array(a, info.n_total, info.n_obj, tail_fill=0.0).reshape(info.n_total, 3)
        out = np.zeros((info.n_total, 3), dtype=np.float32); out[info.n_obj:, 1] = 1.0; return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fi, ep, t = [int(x) for x in self.windows[int(idx)].tolist()]
        info = self.files[fi]; pos, vel = self._get_pos_vel(fi); n_total, n_obj = info.n_total, info.n_obj
        horizon = max(1, min(int(self.rollout_steps), int(info.n_t - 1 - t)))
        x = _torch_f32(pos[ep, t]); v = _torch_f32(vel[ep, t]); xn = _torch_f32(pos[ep, t + 1]); vn = _torch_f32(vel[ep, t + 1])
        x_roll = _torch_f32(pos[ep, t + 1:t + horizon + 1]); v_roll = _torch_f32(vel[ep, t + 1:t + horizon + 1])
        x_init_arr = self._get_array(fi, "x_init")
        if x_init_arr is not None:
            x_init_np = np.asarray(_episode_value(x_init_arr, ep, info.n_ep, pos[ep, 0]), dtype=np.float32)
        else:
            x_init_np = np.asarray(pos[ep, 0], dtype=np.float32)
        types = _torch_i64(info.types); object_id = _torch_i64(info.object_id)
        k_values = self._k_values_for_episode(fi, info, ep)
        ei, rest, coef, edge_oid = self._struct_graph(fi, info, ep, x_init_np, k_values)
        floor_height_arr = self._get_array(fi, "floor_height"); gravity_arr = self._get_array(fi, "gravity"); gravity_on_arr = self._get_array(fi, "gravity_on"); vol_arr = self._get_array(fi, "vol")
        fh = _scalar_float(_episode_value(floor_height_arr, ep, info.n_ep, 0.0), 0.0)
        gravity = np.asarray(_episode_value(gravity_arr, ep, info.n_ep, np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if gravity.size == 1: gravity = np.array([0.0, float(gravity[0]), 0.0], dtype=np.float32)
        if gravity.size < 3: gravity = np.zeros(3, dtype=np.float32)
        gravity = gravity[:3]
        gravity_on = _scalar_float(_episode_value(gravity_on_arr, ep, info.n_ep, float(np.linalg.norm(gravity) > 0)), float(np.linalg.norm(gravity) > 0))
        vol = _fix_particle_array(_episode_value(vol_arr, ep, info.n_ep, np.ones((n_total,), dtype=np.float32)), n_total, n_obj, tail_fill=1.0).reshape(n_total)
        extra_force_arr = self._get_array(fi, "extra_force"); extra_mask_arr = self._get_array(fi, "extra_mask"); ext_force_arr = self._get_array(fi, "ext_force"); ext_mask_arr = self._get_array(fi, "ext_mask")
        gravity_force_arr = self._get_array(fi, "gravity_force"); gravity_mask_arr = self._get_array(fi, "gravity_mask")
        total_force_arr = self._get_array(fi, "total_force")
        if total_force_arr is None:
            total_force_arr = self._get_array(fi, "total_external_force")
        total_mask_arr = self._get_array(fi, "total_mask")
        if total_mask_arr is None:
            total_mask_arr = self._get_array(fi, "total_external_mask")
        def forces_for_frame(tt: int) -> Dict[str, np.ndarray]:
            extra_force = _fix_force_array(_frame_value(extra_force_arr, ep, tt, info.n_ep, info.n_t, np.zeros((n_total, 3), dtype=np.float32)), n_total, n_obj)
            extra_mask = _fix_mask_array(_frame_value(extra_mask_arr, ep, tt, info.n_ep, info.n_t, np.zeros((n_total,), dtype=np.float32)), n_total, n_obj)
            if extra_force_arr is None and ext_force_arr is not None:
                extra_force = _fix_force_array(_frame_value(ext_force_arr, ep, tt, info.n_ep, info.n_t, np.zeros((n_total, 3), dtype=np.float32)), n_total, n_obj)
                extra_mask = _fix_mask_array(_frame_value(ext_mask_arr, ep, tt, info.n_ep, info.n_t, np.zeros((n_total,), dtype=np.float32)), n_total, n_obj)
            gravity_force = _fix_force_array(_frame_value(gravity_force_arr, ep, tt, info.n_ep, info.n_t, np.zeros((n_total, 3), dtype=np.float32)), n_total, n_obj)
            gravity_mask = _fix_mask_array(_frame_value(gravity_mask_arr, ep, tt, info.n_ep, info.n_t, np.zeros((n_total,), dtype=np.float32)), n_total, n_obj)
            if gravity_force_arr is None and gravity_on > 0.5 and np.linalg.norm(gravity) > 0:
                gravity_force = np.zeros((n_total, 3), dtype=np.float32); gravity_force[:n_obj] = vol[:n_obj, None] * gravity[None]
                gravity_mask = np.zeros((n_total,), dtype=np.float32); gravity_mask[:n_obj] = 1.0
            total_force = _fix_force_array(_frame_value(total_force_arr, ep, tt, info.n_ep, info.n_t, extra_force + gravity_force), n_total, n_obj)
            total_mask = _fix_mask_array(_frame_value(total_mask_arr, ep, tt, info.n_ep, info.n_t, ((extra_mask > 0) | (gravity_mask > 0)).astype(np.float32)), n_total, n_obj)
            ext_force = _fix_force_array(_frame_value(ext_force_arr, ep, tt, info.n_ep, info.n_t, extra_force), n_total, n_obj)
            ext_mask = _fix_mask_array(_frame_value(ext_mask_arr, ep, tt, info.n_ep, info.n_t, extra_mask), n_total, n_obj)
            return {"extra_force": extra_force, "extra_mask": extra_mask, "gravity_force": gravity_force, "gravity_mask": gravity_mask, "total_force": total_force, "total_mask": total_mask, "ext_force": ext_force, "ext_mask": ext_mask}
        f0 = forces_for_frame(t); fseq = [forces_for_frame(t + s) for s in range(horizon)]
        def stack(name: str) -> torch.Tensor: return _torch_f32(np.stack([d[name] for d in fseq], axis=0))
        group_idx = _scalar_int(_episode_value(info.group_index, ep, info.n_ep, ep), ep) if info.group_index is not None else ep
        rep_idx = _scalar_int(_episode_value(info.repeat_index, ep, info.n_ep, 0), 0) if info.repeat_index is not None else 0
        kti = _scalar_int(_episode_value(info.k_tuple_index, ep, info.n_ep, 0), 0) if info.k_tuple_index is not None else 0
        return {
            "x": x, "v": v, "x_next": xn, "v_next": vn, "rollout_x_next": x_roll, "rollout_v_next": v_roll, "rollout_steps": torch.tensor(int(horizon), dtype=torch.long),
            "x_init": _torch_f32(x_init_np), "types": types, "object_id": object_id,
            "edge_index_struct": ei.clone(), "rest_lengths": rest.clone(), "edge_coef_k1": coef.clone(), "edge_object_id": edge_oid.clone(),
            "dt": torch.tensor(float(info.dt), dtype=torch.float32), "k_values": _torch_f32(k_values), "k_triplet": _torch_f32(k_values), "k_pair": _torch_f32(k_values), "k": _torch_f32(k_values), "E": _torch_f32(k_values), "k_mean": torch.tensor(float(np.mean(k_values)), dtype=torch.float32),
            "floor_height": torch.tensor(float(fh), dtype=torch.float32), "particle_radius": torch.tensor(float(info.particle_radius), dtype=torch.float32), "vol": _torch_f32(vol), "gravity": _torch_f32(gravity), "gravity_on": torch.tensor(float(gravity_on), dtype=torch.float32),
            "extra_force": _torch_f32(f0["extra_force"]), "extra_mask": _torch_f32(f0["extra_mask"]), "gravity_force": _torch_f32(f0["gravity_force"]), "gravity_mask": _torch_f32(f0["gravity_mask"]), "total_force": _torch_f32(f0["total_force"]), "total_mask": _torch_f32(f0["total_mask"]), "ext_force": _torch_f32(f0["ext_force"]), "ext_mask": _torch_f32(f0["ext_mask"]),
            "rollout_extra_force": stack("extra_force"), "rollout_extra_mask": stack("extra_mask"), "rollout_gravity_force": stack("gravity_force"), "rollout_gravity_mask": stack("gravity_mask"), "rollout_total_force": stack("total_force"), "rollout_total_mask": stack("total_mask"), "rollout_ext_force": stack("ext_force"), "rollout_ext_mask": stack("ext_mask"),
            "env_normals": _torch_f32(self._env_normals(fi, info, ep)), "n_obj": torch.tensor(int(info.n_obj), dtype=torch.long), "n_env": torch.tensor(int(info.n_env), dtype=torch.long), "n_objects": torch.tensor(int(self.n_objects), dtype=torch.long),
            "file_index": torch.tensor(fi, dtype=torch.long), "episode_index": torch.tensor(ep, dtype=torch.long), "time_index": torch.tensor(t, dtype=torch.long), "group_index": torch.tensor(group_idx, dtype=torch.long), "repeat_index": torch.tensor(rep_idx, dtype=torch.long), "k_tuple_index": torch.tensor(kti, dtype=torch.long), "dataset_index": torch.tensor(int(idx), dtype=torch.long),
        }

    def summary(self) -> Dict[str, Any]:
        kvals = []; n_ep = 0
        for fi, info in enumerate(self.files):
            n_ep += info.n_ep
            for ep in range(info.n_ep):
                kvals.append(self._k_values_for_episode(fi, info, ep))
        ka = np.asarray(kvals, dtype=np.float64) if kvals else np.zeros((0, self.n_objects))
        return {"data": self.data_spec, "mode": self.mode, "split": self.split, "split_by": self.split_by, "n_objects": int(self.n_objects), "num_files": len(self.files), "num_episodes_total": int(n_ep), "num_windows": int(len(self.windows)), "num_contact_windows": int(self.window_is_contact.sum()) if self.window_is_contact.size else 0, "contact_window_ratio": float(self.window_is_contact.mean()) if self.window_is_contact.size else 0.0, "requested_rollout_steps": int(self.requested_rollout_steps), "rollout_steps": int(self.rollout_steps), "rollout_was_clipped": bool(self.rollout_steps != self.requested_rollout_steps), "dt_values": sorted({float(f.dt) for f in self.files}), "particle_radius_values": sorted({float(f.particle_radius) for f in self.files}), "k_min": ka.min(axis=0).tolist() if ka.size else None, "k_max": ka.max(axis=0).tolist() if ka.size else None, "k_unique_sample": sorted({tuple(map(float, x)) for x in ka.tolist()})[:20] if ka.size else [], "struct_graph_source": "inferred_from_x_init_and_object_id", "uses_dataset_spring_graph": False, "struct_graph_seed_scheme": self.struct_seed_protocol, "struct_graph_preset": self.struct_graph_preset, "struct_graph_k_obj": int(self.k_obj), "struct_graph_long_k": int(self.struct_long_k), "struct_graph_long_strategy": self.struct_long_strategy, "edge_coef_mode": self.edge_coef_mode, "first_file": self.files[0].path if self.files else None}
