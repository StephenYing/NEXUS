from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import PointTransformerConv as _PointTransformerConv
    from torch_geometric.nn import knn_graph as _pyg_knn_graph
    _HAS_PYG = True
except Exception:
    _PointTransformerConv = None
    _pyg_knn_graph = None
    _HAS_PYG = False


KNN_EDGE_CONVENTION = "receiver_sender_v1"


def validate_checkpoint_knn_edge_convention(checkpoint: Mapping[str, Any]) -> bool:
    """Validate a versioned checkpoint convention; False means legacy/unknown."""
    saved = checkpoint.get("knn_edge_convention")
    if saved is None:
        return False
    if str(saved) != KNN_EDGE_CONVENTION:
        raise ValueError(
            f"Checkpoint KNN edge convention {saved!r} does not match "
            f"runtime convention {KNN_EDGE_CONVENTION!r}"
        )
    return True


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, depth: int = 2, zero_last: bool = False):
        super().__init__()
        layers = []
        d = int(in_dim)
        for _ in range(max(0, int(depth))):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.SiLU())
            d = hidden
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimplePointConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.self_lin = nn.Linear(channels, channels)
        self.msg_lin = nn.Linear(channels, channels)
        self.out_lin = nn.Sequential(nn.SiLU(), nn.Linear(channels, channels))

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.out_lin(self.self_lin(h))
        receiver, sender = edge_index[0], edge_index[1]
        msg = self.msg_lin(h[sender])
        agg = torch.zeros_like(h)
        agg.index_add_(0, receiver, msg)
        deg = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
        deg.index_add_(0, receiver, torch.ones(receiver.numel(), 1, device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        return self.out_lin(self.self_lin(h) + agg)


def _knn_graph_fallback(pos: torch.Tensor, k: int, loop: bool = False) -> torch.Tensor:
    """Return KNN edges as [receiver/query, sender/neighbor]."""
    n = int(pos.size(0))
    if n <= 1 or k <= 0:
        return torch.empty((2, 0), device=pos.device, dtype=torch.long)
    k = min(int(k), n if loop else n - 1)
    d = torch.cdist(pos, pos)
    if not loop:
        d.fill_diagonal_(float("inf"))
    nn = torch.topk(d, k=k, dim=1, largest=False).indices
    row = torch.arange(n, device=pos.device).view(-1, 1).expand(-1, k).reshape(-1)
    col = nn.reshape(-1)
    return torch.stack([row, col], dim=0)


def knn_graph_safe(pos: torch.Tensor, k: int, loop: bool = False, backend: str = "auto") -> torch.Tensor:
    """Return backend-independent KNN edges as [receiver, sender]."""
    if backend not in {"auto", "torch_cdist", "pyg"}:
        raise ValueError(f"Unknown KNN backend: {backend}")
    if pos.size(0) <= 1 or k <= 0:
        return torch.empty((2, 0), device=pos.device, dtype=torch.long)
    if backend == "torch_cdist":
        return _knn_graph_fallback(pos, k=k, loop=loop)
    if backend == "pyg" and not _HAS_PYG:
        raise ImportError("The checkpoint requires the PyG KNN backend")
    if _HAS_PYG:
        try:
            pyg_edge = _pyg_knn_graph(pos, k=min(int(k), int(pos.size(0)) - 1), loop=loop)
            # PyG uses [source/sender, target/receiver]. Convert at the boundary.
            return pyg_edge.flip(0).contiguous()
        except Exception:
            if backend == "pyg":
                raise
    return _knn_graph_fallback(pos, k=k, loop=loop)


def infer_encoder_type_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    """Infer the encoder implementation so legacy checkpoints load exactly."""
    keys = state_dict.keys()
    if any(key.startswith("pt1.self_lin.") for key in keys):
        return "simple"
    if any(key.startswith("pt1.pos_nn.") or key.startswith("pt1.lin_src.") for key in keys):
        return "point_transformer"
    return None


def _scalar_tensor(x: Any, default: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x is None:
        return torch.tensor(float(default), device=device, dtype=dtype)
    if not torch.is_tensor(x):
        x = torch.tensor(x, device=device, dtype=dtype)
    else:
        x = x.to(device=device, dtype=dtype)
    if x.numel() == 0:
        return torch.tensor(float(default), device=device, dtype=dtype)
    return x.reshape(-1)[0]


def _k_values_tensor(k_triplet: Any, k: Any, E: Any, device: torch.device, dtype: torch.dtype, default: float = 100.0, n_objects: int = 3) -> torch.Tensor:
    src = k_triplet if k_triplet is not None else (k if k is not None else E)
    if src is None:
        return torch.full((int(n_objects),), float(default), device=device, dtype=dtype)
    if not torch.is_tensor(src):
        src = torch.tensor(src, device=device, dtype=dtype)
    else:
        src = src.to(device=device, dtype=dtype)
    src = src.reshape(-1)
    n_objects = int(n_objects)
    if src.numel() >= n_objects:
        return src[:n_objects].clamp_min(1e-8)
    if src.numel() == 1:
        return src[0].repeat(n_objects).clamp_min(1e-8)
    out = torch.full((n_objects,), float(default), device=device, dtype=dtype)
    if src.numel() > 0:
        m = min(n_objects, int(src.numel()))
        out[:m] = src[:m]
        if m < n_objects:
            out[m:] = src[:m].mean()
    return out.clamp_min(1e-8)

def _vector3_tensor(x: Any, default: Tuple[float, float, float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x is None:
        return torch.tensor(default, device=device, dtype=dtype).view(1, 3)
    if not torch.is_tensor(x):
        x = torch.tensor(x, device=device, dtype=dtype)
    else:
        x = x.to(device=device, dtype=dtype)
    x = x.reshape(-1)
    if x.numel() == 1:
        return torch.tensor([0.0, float(x.item()), 0.0], device=device, dtype=dtype).view(1, 3)
    if x.numel() < 3:
        return torch.tensor(default, device=device, dtype=dtype).view(1, 3)
    return x[:3].view(1, 3)


def _particle_force(x: Any, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x is None:
        return torch.zeros(n, 3, device=device, dtype=dtype)
    if not torch.is_tensor(x):
        x = torch.tensor(x, device=device, dtype=dtype)
    else:
        x = x.to(device=device, dtype=dtype)
    if x.numel() == 0:
        return torch.zeros(n, 3, device=device, dtype=dtype)
    if x.ndim == 1 and x.numel() == 3:
        return x.view(1, 3).expand(n, 3).clone()
    x = x.reshape(-1, 3)
    if x.size(0) == n:
        return x
    out = torch.zeros(n, 3, device=device, dtype=dtype)
    m = min(n, x.size(0))
    out[:m] = x[:m]
    return out


def _particle_mask(x: Any, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x is None:
        return torch.zeros(n, 1, device=device, dtype=dtype)
    if not torch.is_tensor(x):
        x = torch.tensor(x, device=device, dtype=dtype)
    else:
        x = x.to(device=device, dtype=dtype)
    if x.numel() == 0:
        return torch.zeros(n, 1, device=device, dtype=dtype)
    x = x.reshape(-1, 1)
    if x.size(0) == n:
        return x
    out = torch.zeros(n, 1, device=device, dtype=dtype)
    m = min(n, x.size(0))
    out[:m] = x[:m]
    return out


def _clip_vector_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm is None or float(max_norm) <= 0.0:
        return x
    n = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return x * torch.clamp(float(max_norm) / n, max=1.0)


def _positive_from_zero(z: torch.Tensor) -> torch.Tensor:
    return F.softplus(z) / F.softplus(torch.zeros((), device=z.device, dtype=z.dtype))


class NEXUSDynamics(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        n_objects: int = 3,
        k_knn: int = 12,
        k_pair_contact: int = 4,
        k_floor_contact: int = 8,
        default_particle_radius: float = 0.04,
        k_ref: float = 100.0,
        logk_center: float = 2.0,
        logk_scale: float = 1.0,
        mlp_depth: int = 2,
        coef_detach: bool = True,
        struct_mode: str = "learned_times_k",  # explicit | hybrid | learned_times_k | learned_times_k_geom
        struct_scale: float = 100.0,
        struct_gain_range: float = 0.75,
        use_edge_coef: bool = True,
        struct_k_power: float = 1.0,
        struct_geom_power: float = 1.0,
        struct_min_gain: float = 0.0,
        struct_geom_norm: str = "mean",
        use_struct_damping: bool = True,
        struct_damping_ratio: float = 0.015,
        struct_damping_scale: float = 1.0,
        use_pair_contact: bool = True,
        use_floor_contact: bool = True,
        pair_contact_scale: float = 60000.0,
        floor_contact_scale: float = 80000.0,
        pair_contact_margin_factor: float = 1.0,
        floor_contact_margin_factor: float = 1.0,
        contact_edge_norm: str = "mean_k",
        contact_barrier: str = "relu",
        contact_softplus_beta: float = 20.0,
        contact_cutoff_factor: float = 0.0,
        use_contact_damping: bool = True,
        pair_contact_damping_scale: float = 150.0,
        floor_contact_damping_scale: float = 150.0,
        contact_damping_k_power: float = 0.0,
        damping_param: str = "learned",
        use_global_context: bool = False,
        encoder_type: str = "auto",
        knn_backend: str = "auto",
        external_source: str = "separate",
        external_force_scale: float = 1.0,
        integrator: str = "semi_implicit",
        gravity: Tuple[float, float, float] = (0.0, -4.0, 0.0),
        **_: Any,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.n_objects = int(n_objects)
        self.k_knn = int(k_knn)
        self.k_pair_contact = int(k_pair_contact)
        self.k_floor_contact = int(k_floor_contact)
        self.default_particle_radius = float(default_particle_radius)
        self.k_ref = float(k_ref)
        self.logk_center = float(logk_center)
        self.logk_scale = float(logk_scale)
        self.mlp_depth = int(mlp_depth)
        self.coef_detach = bool(coef_detach)
        self.struct_mode = str(struct_mode)
        self.struct_scale = float(struct_scale)
        self.struct_gain_range = float(struct_gain_range)
        self.use_edge_coef = bool(use_edge_coef)
        self.struct_k_power = float(struct_k_power)
        self.struct_geom_power = float(struct_geom_power)
        self.struct_min_gain = float(struct_min_gain)
        self.struct_geom_norm = str(struct_geom_norm or "mean")
        self.use_struct_damping = bool(use_struct_damping)
        self.struct_damping_ratio = float(struct_damping_ratio)
        self.struct_damping_scale = float(struct_damping_scale)
        self.use_pair_contact = bool(use_pair_contact)
        self.use_floor_contact = bool(use_floor_contact)
        self.pair_contact_scale = float(pair_contact_scale)
        self.floor_contact_scale = float(floor_contact_scale)
        self.pair_contact_margin_factor = float(pair_contact_margin_factor)
        self.floor_contact_margin_factor = float(floor_contact_margin_factor)
        self.contact_edge_norm = str(contact_edge_norm)
        self.contact_barrier = str(contact_barrier)
        self.contact_softplus_beta = float(contact_softplus_beta)
        self.contact_cutoff_factor = float(contact_cutoff_factor)
        self.use_contact_damping = bool(use_contact_damping)
        self.pair_contact_damping_scale = float(pair_contact_damping_scale)
        self.floor_contact_damping_scale = float(floor_contact_damping_scale)
        self.contact_damping_k_power = float(contact_damping_k_power)
        self.damping_param = str(damping_param)
        self.use_global_context = bool(use_global_context)
        self.encoder_type = str(encoder_type or "auto").lower()
        self.knn_backend = str(knn_backend)
        if self.knn_backend not in {"auto", "torch_cdist", "pyg"}:
            raise ValueError(f"Unknown KNN backend: {self.knn_backend}")
        if self.encoder_type in ("point_transformer", "pt", "pyg") and not _HAS_PYG:
            raise ImportError("PointTransformer was explicitly requested but torch_geometric is unavailable; install the declared dependencies or select --encoder_type simple explicitly")
        self.external_source = str(external_source)
        self.external_force_scale = float(external_force_scale)
        self.integrator = str(integrator)
        self.register_buffer("default_g_vec", torch.tensor(gravity, dtype=torch.float32).view(1, 3))

        self.node_feature_dim = 14 + self.n_objects
        self.node_in = nn.Linear(self.node_feature_dim, hidden)
        use_pt = (_HAS_PYG and self.encoder_type in ("auto", "point_transformer", "pt", "pyg"))
        if use_pt:
            try:
                self.pt1 = _PointTransformerConv(in_channels=hidden, out_channels=hidden)
                self.pt2 = _PointTransformerConv(in_channels=hidden, out_channels=hidden)
                self.encoder_type_actual = "point_transformer"
            except Exception:
                if self.encoder_type != "auto":
                    raise
                self.pt1 = SimplePointConv(hidden)
                self.pt2 = SimplePointConv(hidden)
                self.encoder_type_actual = "simple"
        else:
            self.pt1 = SimplePointConv(hidden)
            self.pt2 = SimplePointConv(hidden)
            self.encoder_type_actual = "simple"
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.global_context_net = MLP(2 * hidden + self.n_objects, hidden, hidden=hidden, depth=1, zero_last=True)

        edge_in = 2 * hidden + 5  # strain, rest, edge_coef, logk_edge, relv
        self.struct_gain_net = MLP(edge_in, 1, hidden=hidden, depth=mlp_depth, zero_last=True)
        self.struct_k_net = MLP(edge_in, 1, hidden=hidden, depth=mlp_depth, zero_last=True)
        contact_in = 2 * hidden + 5  # metric, delta, relv, logk_row, logk_col/zero
        self.contact_k_net = MLP(contact_in, 1, hidden=hidden, depth=mlp_depth, zero_last=True)
        self.contact_c_net = MLP(contact_in, 1, hidden=hidden, depth=mlp_depth, zero_last=True)

    def _logk_norm_values(self, k_triplet: torch.Tensor) -> torch.Tensor:
        return (torch.log10(k_triplet.clamp_min(1e-8)) - self.logk_center) / max(self.logk_scale, 1e-8)

    def _node_k_values(self, object_id: torch.Tensor, types: torch.Tensor, k_triplet: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(types.numel(), device=types.device, dtype=k_triplet.dtype)
        for oid in range(self.n_objects):
            m = (types == 0) & (object_id == oid)
            if oid < k_triplet.numel():
                out[m] = k_triplet[oid]
        return out

    def _encode(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        types: torch.Tensor,
        object_id: torch.Tensor,
        x_init: torch.Tensor,
        k_triplet: torch.Tensor,
        vol: Optional[torch.Tensor],
        gravity_on: torch.Tensor,
        extra_force: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n = x.size(0)
        device, dtype = x.device, x.dtype
        if vol is None:
            mass = torch.ones(n, 1, device=device, dtype=dtype)
        else:
            mass = vol.to(device=device, dtype=dtype).reshape(-1, 1)
            if mass.size(0) != n:
                fixed = torch.ones(n, 1, device=device, dtype=dtype)
                fixed[: min(n, mass.size(0))] = mass[: min(n, mass.size(0))]
                mass = fixed
        ef = _particle_force(extra_force, n, device, dtype)
        log_mass = torch.log(mass.clamp_min(1e-8))
        node_k = self._node_k_values(object_id, types, k_triplet).clamp_min(1e-8)
        logk_node = ((torch.log10(node_k) - self.logk_center) / max(self.logk_scale, 1e-8)).view(-1, 1)
        logk_node = torch.where((types == 0).view(-1, 1), logk_node, torch.zeros_like(logk_node))
        onehot_parts = [((object_id == oid) & (types == 0)).to(dtype) for oid in range(self.n_objects)]
        onehot_parts.append((types != 0).to(dtype))
        onehot = torch.stack(onehot_parts, dim=-1)
        g_on = gravity_on.view(1, 1).to(device=device, dtype=dtype).expand(n, 1)
        feat = torch.cat([
            x - x_init,
            v,
            types.float().view(-1, 1),
            onehot,
            log_mass,
            logk_node,
            g_on,
            ef,
        ], dim=-1)
        h = self.node_in(feat)
        enc_edge = knn_graph_safe(x.detach() if self.coef_detach else x, self.k_knn, loop=False, backend=self.knn_backend)
        # PointTransformerConv follows PyG's [sender, receiver] convention.
        conv_edge = enc_edge.flip(0).contiguous() if self.encoder_type_actual == "point_transformer" else enc_edge
        h1 = self.pt1(h, x.detach() if self.coef_detach else x, conv_edge)
        h = self.norm1(h + h1)
        h2 = self.pt2(h, x.detach() if self.coef_detach else x, conv_edge)
        h = self.norm2(h + h2)
        if self.use_global_context and n > 0:
            dyn = (types == 0)
            h_dyn = h[dyn] if bool(dyn.any()) else h
            g_mean = h_dyn.mean(dim=0, keepdim=True)
            g_max = h_dyn.max(dim=0, keepdim=True).values
            logk_vals = self._logk_norm_values(k_triplet).view(1, self.n_objects)
            g = self.global_context_net(torch.cat([g_mean, g_max, logk_vals], dim=-1))
            h = h + g.expand_as(h)
        return h

    def _barrier_phi(self, delta: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
        mode = (self.contact_barrier or "relu").lower()
        if mode in ("relu", "hinge"):
            return F.relu(delta)
        beta = torch.tensor(float(self.contact_softplus_beta), device=delta.device, dtype=delta.dtype).clamp_min(1e-6)
        if mode in ("softplus_raw", "raw"):
            return F.softplus(beta * delta) / beta
        scale = margin.to(device=delta.device, dtype=delta.dtype).clamp_min(1e-8)
        return scale * F.softplus(beta * delta / scale) / beta

    def _edge_weights(self, row: torch.Tensor, k: int, dtype: torch.dtype) -> torch.Tensor:
        if row.numel() == 0:
            return torch.empty(0, device=row.device, dtype=dtype)
        mode = (self.contact_edge_norm or "none").lower()
        if mode in ("mean_k", "k", "divide_k"):
            return torch.full((row.numel(),), 1.0 / max(1, int(k)), device=row.device, dtype=dtype)
        if mode in ("mean_obj", "object"):
            uniq, inv = torch.unique(row, return_inverse=True)
            cnt = torch.zeros(uniq.numel(), device=row.device, dtype=dtype)
            cnt.index_add_(0, inv, torch.ones_like(inv, dtype=dtype))
            return 1.0 / cnt[inv].clamp_min(1.0)
        return torch.ones(row.numel(), device=row.device, dtype=dtype)

    def _pair_contact_edges(self, x: torch.Tensor, types: torch.Tensor, object_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Dynamic object-object contact graph over all unordered object pairs."""
        if self.k_pair_contact <= 0:
            empty = torch.empty(0, device=x.device, dtype=torch.long)
            return empty, empty, 0
        rows, cols = [], []
        maxk = 0
        for a in range(self.n_objects):
            idx_a = torch.where((types == 0) & (object_id == a))[0]
            if idx_a.numel() == 0:
                continue
            for b in range(a + 1, self.n_objects):
                idx_b = torch.where((types == 0) & (object_id == b))[0]
                if idx_b.numel() == 0:
                    continue
                d_ab = torch.cdist(x[idx_a], x[idx_b])
                k_ab = min(int(self.k_pair_contact), int(idx_b.numel()))
                if k_ab > 0:
                    _, nn_ab = torch.topk(d_ab, k=k_ab, dim=1, largest=False)
                    rows.append(idx_a.view(-1, 1).expand(-1, k_ab).reshape(-1))
                    cols.append(idx_b[nn_ab.reshape(-1)])
                    maxk = max(maxk, k_ab)
                k_ba = min(int(self.k_pair_contact), int(idx_a.numel()))
                if k_ba > 0:
                    _, nn_ba = torch.topk(d_ab.T, k=k_ba, dim=1, largest=False)
                    rows.append(idx_b.view(-1, 1).expand(-1, k_ba).reshape(-1))
                    cols.append(idx_a[nn_ba.reshape(-1)])
                    maxk = max(maxk, k_ba)
        if not rows:
            empty = torch.empty(0, device=x.device, dtype=torch.long)
            return empty, empty, 0
        return torch.cat(rows, dim=0), torch.cat(cols, dim=0), maxk

    def _floor_contact_edges(self, x: torch.Tensor, types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        obj = torch.where(types == 0)[0]
        env = torch.where(types != 0)[0]
        if obj.numel() == 0 or env.numel() == 0 or self.k_floor_contact <= 0:
            empty = torch.empty(0, device=x.device, dtype=torch.long)
            return empty, empty, 0
        k = min(int(self.k_floor_contact), int(env.numel()))
        d = torch.cdist(x[obj], x[env])
        _, nn = torch.topk(d, k=k, dim=1, largest=False)
        row = obj.view(-1, 1).expand(-1, k).reshape(-1)
        col = env[nn.reshape(-1)]
        return row, col, k

    def _external_force_tensor(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
        types: torch.Tensor,
        vol: torch.Tensor,
        gravity: Any,
        gravity_on: Any,
        extra_force: Any,
        extra_mask: Any,
        gravity_force: Any,
        gravity_mask: Any,
        total_force: Any,
        total_mask: Any,
        ext_force: Any,
        ext_mask: Any,
    ) -> torch.Tensor:
        source = (self.external_source or "separate").lower()
        if source == "none":
            return torch.zeros(n, 3, device=device, dtype=dtype)
        if source in ("total", "total_force"):
            return _particle_force(total_force, n, device, dtype) * _particle_mask(total_mask, n, device, dtype)
        if source in ("legacy", "ext"):
            return _particle_force(ext_force, n, device, dtype) * _particle_mask(ext_mask, n, device, dtype)
        if source == "extra":
            return _particle_force(extra_force, n, device, dtype) * _particle_mask(extra_mask, n, device, dtype)
        if source == "gravity":
            ef = torch.zeros(n, 3, device=device, dtype=dtype)
        else:
            ef = _particle_force(extra_force, n, device, dtype) * _particle_mask(extra_mask, n, device, dtype)
        gf = _particle_force(gravity_force, n, device, dtype)
        gm = _particle_mask(gravity_mask, n, device, dtype)
        if gf.abs().sum() == 0:
            g = _vector3_tensor(gravity, tuple(self.default_g_vec.reshape(-1).tolist()), device, dtype)
            g_on = _scalar_tensor(gravity_on, 1.0, device, dtype)
            dyn = (types == 0).to(dtype).view(-1, 1)
            gf = vol.reshape(-1, 1) * g * dyn * g_on
            gm = dyn * g_on
        return ef + gf * gm

    def _energy_and_dissipation(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        *,
        types: torch.Tensor,
        object_id: torch.Tensor,
        x_init: torch.Tensor,
        edge_index_struct: torch.Tensor,
        rest_lengths: torch.Tensor,
        edge_coef_k1: Optional[torch.Tensor],
        edge_object_id: Optional[torch.Tensor],
        k_triplet: Any = None,
        k: Any = None,
        E: Any = None,
        vol: Any = None,
        particle_radius: Any = None,
        env_normals: Any = None,
        gravity: Any = None,
        gravity_on: Any = None,
        extra_force: Any = None,
        extra_mask: Any = None,
        gravity_force: Any = None,
        gravity_mask: Any = None,
        total_force: Any = None,
        total_mask: Any = None,
        ext_force: Any = None,
        ext_mask: Any = None,
        return_tensors: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        device, dtype = x.device, x.dtype
        n = x.size(0)
        types = types.to(device=device, dtype=torch.long)
        object_id = object_id.to(device=device, dtype=torch.long)
        x_init = x_init.to(device=device, dtype=dtype)
        k_triplet_t = _k_values_tensor(k_triplet, k, E, device, dtype, default=self.k_ref, n_objects=self.n_objects)
        g_on = _scalar_tensor(gravity_on, 1.0, device, dtype)
        if vol is None:
            mass = torch.ones(n, device=device, dtype=dtype)
        else:
            mass = vol.to(device=device, dtype=dtype).reshape(-1)
            if mass.numel() != n:
                fixed = torch.ones(n, device=device, dtype=dtype)
                fixed[: min(n, mass.numel())] = mass[: min(n, mass.numel())]
                mass = fixed
            mass = mass.clamp_min(1e-8)

        h = self._encode(
            x.detach() if self.coef_detach else x,
            v.detach() if self.coef_detach else v,
            types,
            object_id,
            x_init,
            k_triplet_t,
            mass,
            g_on,
            extra_force,
        )
        h_coef = h
        x_coef = x.detach() if self.coef_detach else x
        v_coef = v.detach() if self.coef_detach else v
        U = torch.zeros((), device=device, dtype=dtype)
        R = torch.zeros((), device=device, dtype=dtype)
        diag: Dict[str, Any] = {}

        # Per-object structural energy on inferred edges.
        U_struct = torch.zeros((), device=device, dtype=dtype)
        R_struct = torch.zeros((), device=device, dtype=dtype)
        if edge_index_struct is not None and edge_index_struct.numel() > 0:
            edge_index_struct = edge_index_struct.to(device=device, dtype=torch.long)
            row_s, col_s = edge_index_struct[0], edge_index_struct[1]
            rest = rest_lengths.to(device=device, dtype=dtype).reshape(-1).clamp_min(1e-8)
            if edge_coef_k1 is None:
                edge_coef = 1.0 / rest
            else:
                edge_coef = edge_coef_k1.to(device=device, dtype=dtype).reshape(-1)
                if edge_coef.numel() != rest.numel():
                    edge_coef = 1.0 / rest
                edge_coef = edge_coef.clamp_min(0.0)
            if edge_object_id is None:
                eo = object_id[row_s].clamp(min=0, max=self.n_objects-1)
            else:
                eo = edge_object_id.to(device=device, dtype=torch.long).reshape(-1)
                if eo.numel() != rest.numel():
                    eo = object_id[row_s].clamp(min=0, max=self.n_objects-1)
            eo = eo.clamp(min=0, max=self.n_objects-1)
            k_edge = k_triplet_t[eo].clamp_min(1e-8)
            logk_norm_edge = ((torch.log10(k_edge) - self.logk_center) / max(self.logk_scale, 1e-8))
            dx = x[row_s] - x[col_s]
            dist = torch.norm(dx, dim=-1).clamp_min(1e-8)
            stretch = dist - rest
            direction = dx / dist.unsqueeze(-1)
            relv = torch.sum((v[row_s] - v[col_s]) * direction, dim=-1)
            dx_c = x_coef[row_s] - x_coef[col_s]
            dist_c = torch.norm(dx_c, dim=-1).clamp_min(1e-8)
            strain_c = (dist_c - rest) / rest
            dir_c = dx_c / dist_c.unsqueeze(-1)
            relv_c = torch.sum((v_coef[row_s] - v_coef[col_s]) * dir_c, dim=-1)
            edge_feat = torch.cat([
                h_coef[row_s],
                h_coef[col_s],
                strain_c.unsqueeze(-1),
                rest.unsqueeze(-1),
                edge_coef.unsqueeze(-1),
                logk_norm_edge.unsqueeze(-1),
                relv_c.unsqueeze(-1),
            ], dim=-1)
            base = k_edge * (edge_coef if self.use_edge_coef else torch.ones_like(edge_coef))
            mode = (self.struct_mode or "learned_times_k").lower()
            if mode == "explicit":
                k_eff = self.struct_scale * base
            elif mode == "hybrid":
                raw_gain = self.struct_gain_net(edge_feat).squeeze(-1)
                gain = torch.exp(self.struct_gain_range * torch.tanh(raw_gain))
                k_eff = self.struct_scale * base * gain
            elif mode in ("learned_times_k", "learned_k"):
                k_factor = (k_edge / max(self.k_ref, 1e-8)).clamp_min(1e-8).pow(float(self.struct_k_power))
                k_eff = self.struct_scale * k_factor * _positive_from_zero(self.struct_k_net(edge_feat).squeeze(-1))
            elif mode in ("learned_times_k_geom", "learned_geom_k", "ltk_geom"):
                learned_gain = _positive_from_zero(self.struct_k_net(edge_feat).squeeze(-1))
                geom = edge_coef.clamp_min(1e-8) if self.use_edge_coef else torch.ones_like(edge_coef)
                norm_mode = (self.struct_geom_norm or "mean").lower()
                if norm_mode in ("mean", "avg", "average"):
                    geom = geom / geom.detach().mean().clamp_min(1e-8)
                elif norm_mode == "median":
                    geom = geom / geom.detach().median().clamp_min(1e-8)
                geom_factor = geom.clamp_min(1e-8).pow(float(self.struct_geom_power))
                k_factor = (k_edge / max(self.k_ref, 1e-8)).clamp_min(1e-8).pow(float(self.struct_k_power))
                k_eff = self.struct_scale * k_factor * geom_factor * (learned_gain + float(self.struct_min_gain))
            else:
                k_eff = self.struct_scale * _positive_from_zero(self.struct_k_net(edge_feat).squeeze(-1))
            k_eff = k_eff.clamp_min(0.0)
            U_struct = torch.sum(0.5 * k_eff * stretch.pow(2))
            U = U + U_struct
            if self.use_struct_damping and self.struct_damping_ratio > 0 and self.struct_damping_scale > 0:
                c_eff = float(self.struct_damping_scale) * float(self.struct_damping_ratio) * 2.0 * torch.sqrt(k_eff.detach().clamp_min(1e-12))
                R_struct = torch.sum(0.5 * c_eff * relv.pow(2))
                R = R + R_struct
        diag["U_struct_tensor"] = U_struct
        diag["R_struct_tensor"] = R_struct

        U_pair = torch.zeros((), device=device, dtype=dtype)
        R_pair = torch.zeros((), device=device, dtype=dtype)
        if self.use_pair_contact:
            row, col, kk = self._pair_contact_edges(x_coef, types, object_id)
            if row.numel() > 0:
                pr = _scalar_tensor(particle_radius, self.default_particle_radius, device, dtype).clamp_min(1e-8)
                margin = 2.0 * pr * float(self.pair_contact_margin_factor)
                dx = x[row] - x[col]
                dx_c = x_coef[row] - x_coef[col]
                dist = torch.norm(dx, dim=-1).clamp_min(1e-8)
                dist_c = torch.norm(dx_c, dim=-1).clamp_min(1e-8)
                nrm = dx / dist.unsqueeze(-1)
                delta = margin - dist
                relv_n = torch.sum((v[row] - v[col]) * nrm, dim=-1)
                if self.contact_cutoff_factor and self.contact_cutoff_factor > 0:
                    keep = delta.detach() > (1.0 - float(self.contact_cutoff_factor)) * margin
                    row, col, delta, relv_n, dist_c = row[keep], col[keep], delta[keep], relv_n[keep], dist_c[keep]
                if row.numel() > 0:
                    phi = self._barrier_phi(delta, margin)
                    node_k = self._node_k_values(object_id, types, k_triplet_t).clamp_min(1e-8)
                    logkr = ((torch.log10(node_k[row]) - self.logk_center) / max(self.logk_scale, 1e-8))
                    logkc = ((torch.log10(node_k[col]) - self.logk_center) / max(self.logk_scale, 1e-8))
                    contact_feat = torch.cat([
                        h_coef[row], h_coef[col],
                        dist_c.unsqueeze(-1),
                        delta.detach().unsqueeze(-1),
                        relv_n.detach().unsqueeze(-1),
                        logkr.unsqueeze(-1),
                        logkc.unsqueeze(-1),
                    ], dim=-1)
                    k_c = float(self.pair_contact_scale) * _positive_from_zero(self.contact_k_net(contact_feat).squeeze(-1))
                    w = 0.5 * self._edge_weights(row, kk, dtype)  # bidirectional pair graph roughly double-counts
                    U_pair = torch.sum(w * 0.5 * k_c * phi.pow(2))
                    U = U + U_pair
                    if self.use_contact_damping and self.pair_contact_damping_scale > 0:
                        if (self.damping_param or "learned").lower() == "constant":
                            c_c = torch.full_like(phi, float(self.pair_contact_damping_scale))
                        else:
                            c_c = float(self.pair_contact_damping_scale) * _positive_from_zero(self.contact_c_net(contact_feat).squeeze(-1))
                        if self.contact_damping_k_power != 0:
                            # Use average object stiffness as optional context scaling.
                            k_avg = torch.sqrt((node_k[row] * node_k[col]).clamp_min(1e-8))
                            c_c = c_c * (k_avg / max(self.k_ref, 1e-8)).clamp_min(1e-8).pow(float(self.contact_damping_k_power))
                        v_app = F.relu(-relv_n)
                        active = (phi.detach() > 0).to(dtype)
                        R_pair = torch.sum(w * active * 0.5 * c_c * v_app.pow(2))
                        R = R + R_pair
        diag["U_pair_contact_tensor"] = U_pair
        diag["R_pair_contact_tensor"] = R_pair

        U_floor = torch.zeros((), device=device, dtype=dtype)
        R_floor = torch.zeros((), device=device, dtype=dtype)
        if self.use_floor_contact:
            row, col, kk = self._floor_contact_edges(x_coef, types)
            if row.numel() > 0:
                pr = _scalar_tensor(particle_radius, self.default_particle_radius, device, dtype).clamp_min(1e-8)
                margin = pr * float(self.floor_contact_margin_factor)
                normals = _particle_force(env_normals, n, device, dtype) if env_normals is not None else torch.zeros(n, 3, device=device, dtype=dtype)
                normals = normals / torch.norm(normals, dim=-1, keepdim=True).clamp_min(1e-8)
                n_plane = normals[col]
                dx = x[row] - x[col]
                dx_c = x_coef[row] - x_coef[col]
                gap = torch.sum(dx * n_plane, dim=-1)
                gap_c = torch.sum(dx_c * n_plane, dim=-1)
                delta = margin - gap
                relv_n = torch.sum((v[row] - v[col]) * n_plane, dim=-1)
                if self.contact_cutoff_factor and self.contact_cutoff_factor > 0:
                    keep = delta.detach() > (1.0 - float(self.contact_cutoff_factor)) * margin
                    row, col, delta, relv_n, gap_c = row[keep], col[keep], delta[keep], relv_n[keep], gap_c[keep]
                if row.numel() > 0:
                    phi = self._barrier_phi(delta, margin)
                    node_k = self._node_k_values(object_id, types, k_triplet_t).clamp_min(1e-8)
                    logkr = ((torch.log10(node_k[row]) - self.logk_center) / max(self.logk_scale, 1e-8))
                    logkzero = torch.zeros_like(logkr)
                    contact_feat = torch.cat([
                        h_coef[row], h_coef[col],
                        gap_c.unsqueeze(-1),
                        delta.detach().unsqueeze(-1),
                        relv_n.detach().unsqueeze(-1),
                        logkr.unsqueeze(-1),
                        logkzero.unsqueeze(-1),
                    ], dim=-1)
                    k_c = float(self.floor_contact_scale) * _positive_from_zero(self.contact_k_net(contact_feat).squeeze(-1))
                    w = self._edge_weights(row, kk, dtype)
                    U_floor = torch.sum(w * 0.5 * k_c * phi.pow(2))
                    U = U + U_floor
                    if self.use_contact_damping and self.floor_contact_damping_scale > 0:
                        if (self.damping_param or "learned").lower() == "constant":
                            c_c = torch.full_like(phi, float(self.floor_contact_damping_scale))
                        else:
                            c_c = float(self.floor_contact_damping_scale) * _positive_from_zero(self.contact_c_net(contact_feat).squeeze(-1))
                        if self.contact_damping_k_power != 0:
                            c_c = c_c * (node_k[row] / max(self.k_ref, 1e-8)).clamp_min(1e-8).pow(float(self.contact_damping_k_power))
                        v_app = F.relu(-relv_n)
                        active = (phi.detach() > 0).to(dtype)
                        R_floor = torch.sum(w * active * 0.5 * c_c * v_app.pow(2))
                        R = R + R_floor
        diag["U_floor_contact_tensor"] = U_floor
        diag["R_floor_contact_tensor"] = R_floor

        f_ext = self._external_force_tensor(
            n=n,
            device=device,
            dtype=dtype,
            types=types,
            vol=mass,
            gravity=gravity,
            gravity_on=gravity_on,
            extra_force=extra_force,
            extra_mask=extra_mask,
            gravity_force=gravity_force,
            gravity_mask=gravity_mask,
            total_force=total_force,
            total_mask=total_mask,
            ext_force=ext_force,
            ext_mask=ext_mask,
        )
        U_ext = -float(self.external_force_scale) * torch.sum(f_ext * x) if self.external_force_scale != 0 else torch.zeros((), device=device, dtype=dtype)
        U = U + U_ext
        diag["U_ext_tensor"] = U_ext
        return U, R, diag

    def _single_step(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        *,
        types: torch.Tensor,
        object_id: torch.Tensor,
        dt: float,
        edge_index_struct: torch.Tensor,
        rest_lengths: torch.Tensor,
        edge_coef_k1: Optional[torch.Tensor],
        edge_object_id: Optional[torch.Tensor],
        x_init: torch.Tensor,
        n_substeps: int = 1,
        force_clip: float = 0.0,
        acc_clip: float = 0.0,
        return_diagnostics: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        device, dtype = x.device, x.dtype
        types = types.to(device=device, dtype=torch.long)
        object_id = object_id.to(device=device, dtype=torch.long)
        n_sub = max(1, int(n_substeps))
        hdt = float(dt) / float(n_sub)
        x_cur, v_cur = x, v
        last_diag: Dict[str, Any] = {}
        for _ in range(n_sub):
            with torch.enable_grad():
                x_req = x_cur.detach().requires_grad_(True)
                v_req = v_cur.detach().requires_grad_(True)
                U, R, diag = self._energy_and_dissipation(
                    x_req,
                    v_req,
                    types=types,
                    object_id=object_id,
                    x_init=x_init,
                    edge_index_struct=edge_index_struct,
                    rest_lengths=rest_lengths,
                    edge_coef_k1=edge_coef_k1,
                    edge_object_id=edge_object_id,
                    return_tensors=return_diagnostics,
                    **kwargs,
                )
                need_graph = self.training
                FU = -torch.autograd.grad(U, x_req, create_graph=need_graph, retain_graph=True, allow_unused=False)[0]
                if R.requires_grad:
                    FR = -torch.autograd.grad(R, v_req, create_graph=need_graph, retain_graph=need_graph, allow_unused=True)[0]
                    if FR is None:
                        FR = torch.zeros_like(v_req)
                else:
                    FR = torch.zeros_like(v_req)
                force = _clip_vector_norm(FU + FR, force_clip)
                vol = kwargs.get("vol", None)
                if vol is None:
                    mass = torch.ones(x.size(0), 1, device=device, dtype=dtype)
                else:
                    mass = vol.to(device=device, dtype=dtype).reshape(-1, 1)
                    if mass.size(0) != x.size(0):
                        fixed = torch.ones(x.size(0), 1, device=device, dtype=dtype)
                        fixed[: min(x.size(0), mass.size(0))] = mass[: min(x.size(0), mass.size(0))]
                        mass = fixed
                    mass = mass.clamp_min(1e-8)
                dynamic = (types == 0).to(dtype).view(-1, 1)
                acc = force / mass * dynamic
                acc = _clip_vector_norm(acc, acc_clip)
                if (self.integrator or "semi_implicit").lower() in ("explicit", "forward"):
                    x_next_dyn = x_req + v_req * hdt
                    v_next_dyn = v_req + acc * hdt
                else:
                    v_next_dyn = v_req + acc * hdt
                    x_next_dyn = x_req + v_next_dyn * hdt
                x_cur = torch.where(dynamic.bool(), x_next_dyn, x_req.detach())
                v_cur = torch.where(dynamic.bool(), v_next_dyn, torch.zeros_like(v_req))
                last_diag = diag
        return x_cur, v_cur, last_diag

    def forward(self, x: torch.Tensor, v: torch.Tensor, types: torch.Tensor, dt: Any, edge_index_struct: torch.Tensor, rest_lengths: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        dt_float = float(dt.detach().cpu().item()) if torch.is_tensor(dt) else float(dt)
        n_substeps = int(kwargs.pop("n_substeps", 1))
        force_clip = float(kwargs.pop("force_clip", 0.0))
        acc_clip = float(kwargs.pop("acc_clip", 0.0))
        object_id = kwargs.pop("object_id")
        x_init = kwargs.pop("x_init")
        edge_coef_k1 = kwargs.pop("edge_coef_k1", None)
        edge_object_id = kwargs.pop("edge_object_id", None)
        xp, vp, _ = self._single_step(
            x, v,
            types=types,
            object_id=object_id,
            dt=dt_float,
            edge_index_struct=edge_index_struct,
            rest_lengths=rest_lengths,
            edge_coef_k1=edge_coef_k1,
            edge_object_id=edge_object_id,
            x_init=x_init,
            n_substeps=n_substeps,
            force_clip=force_clip,
            acc_clip=acc_clip,
            return_diagnostics=False,
            **kwargs,
        )
        return xp, vp

    def forward_with_diagnostics(self, x: torch.Tensor, v: torch.Tensor, types: torch.Tensor, dt: Any, edge_index_struct: torch.Tensor, rest_lengths: torch.Tensor, **kwargs: Any):
        dt_float = float(dt.detach().cpu().item()) if torch.is_tensor(dt) else float(dt)
        n_substeps = int(kwargs.pop("n_substeps", 1))
        force_clip = float(kwargs.pop("force_clip", 0.0))
        acc_clip = float(kwargs.pop("acc_clip", 0.0))
        object_id = kwargs.pop("object_id")
        x_init = kwargs.pop("x_init")
        edge_coef_k1 = kwargs.pop("edge_coef_k1", None)
        edge_object_id = kwargs.pop("edge_object_id", None)
        xp, vp, diag = self._single_step(
            x, v,
            types=types,
            object_id=object_id,
            dt=dt_float,
            edge_index_struct=edge_index_struct,
            rest_lengths=rest_lengths,
            edge_coef_k1=edge_coef_k1,
            edge_object_id=edge_object_id,
            x_init=x_init,
            n_substeps=n_substeps,
            force_clip=force_clip,
            acc_clip=acc_clip,
            return_diagnostics=True,
            **kwargs,
        )
        out: Dict[str, float] = {}
        for k, val in diag.items():
            if torch.is_tensor(val):
                out[k.replace("_tensor", "")] = float(val.detach().cpu().reshape(-1)[0]) if val.numel() else 0.0
        return xp, vp, out



def make_model(model_family: str = "nexus", **kwargs: Any) -> nn.Module:
    """Build NEXUS; accept historical NEXUS identifiers for saved checkpoints."""
    family = str(model_family or "nexus").lower()
    if family not in ("nexus", "enerctrl", "nsef", "energy", "energy_dissipation"):
        raise ValueError(f"Unsupported model family {family!r}; this release contains NEXUS only")
    return NEXUSDynamics(**kwargs)
