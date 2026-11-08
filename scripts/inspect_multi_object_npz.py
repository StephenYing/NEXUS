#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, os
from typing import List
import numpy as np


def discover(spec: str) -> List[str]:
    if os.path.isdir(spec):
        return sorted(glob.glob(os.path.join(spec, '*.npz')))
    if any(c in spec for c in '*?[]'):
        return sorted(glob.glob(spec))
    return [spec]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--max_files', type=int, default=8)
    ap.add_argument('--max_eps_per_file', type=int, default=32)
    ap.add_argument('--min_stored_states', type=int, default=49,
                    help='Minimum trajectory states; set 0 explicitly only when inspecting legacy data')
    args = ap.parse_args()
    files = discover(args.data)[:args.max_files]
    if not files:
        raise FileNotFoundError(args.data)
    rows = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            pos = np.asarray(z['pos'], dtype=np.float32)
            vel = np.asarray(z['vel'], dtype=np.float32) if 'vel' in z else None
            n_ep, T, N, _ = pos.shape
            if T < args.min_stored_states:
                raise ValueError(f'{f}: found {T} stored states; expected at least {args.min_stored_states} for true H48')
            n = min(n_ep, args.max_eps_per_file)
            dt = float(np.asarray(z['dt']).reshape(-1)[0]) if 'dt' in z else 1.0
            pr = float(np.asarray(z['particle_radius']).reshape(-1)[0]) if 'particle_radius' in z else 0.04
            types = np.asarray(z['types'], dtype=np.int64).reshape(-1) if 'types' in z else np.zeros(N, dtype=np.int64)
            oid = np.asarray(z['object_id'], dtype=np.int64).reshape(-1) if 'object_id' in z else np.zeros(N, dtype=np.int64)
            n_objects = int(np.asarray(z['n_objects']).reshape(-1)[0]) if 'n_objects' in z else int(max(1, oid[oid>=0].max()+1 if np.any(oid>=0) else 1))
            vel_err = np.nan
            if vel is not None and T > 1:
                pred = pos[:n,1:] - pos[:n,:-1] - vel[:n,1:] * dt
                vel_err = float(np.linalg.norm(pred, axis=-1).mean())
            pair_contacts = []
            pair_pens = []
            for ep in range(n):
                mx = 0.0; active = False
                for a in range(n_objects):
                    ia = np.where((types == 0) & (oid == a))[0]
                    for b in range(a+1, n_objects):
                        ib = np.where((types == 0) & (oid == b))[0]
                        if ia.size == 0 or ib.size == 0: continue
                        for t in range(T):
                            d = np.linalg.norm(pos[ep,t,ia][:,None,:] - pos[ep,t,ib][None,:,:], axis=-1)
                            pen = np.maximum(2*pr - d, 0.0)
                            if pen.max() > 0: active = True
                            mx = max(mx, float(pen.max()))
                pair_contacts.append(float(active)); pair_pens.append(mx)
            floor_pen = []
            if np.any(types != 0):
                dyn = types == 0
                fh = float(np.asarray(z['floor_height']).reshape(-1)[0]) if 'floor_height' in z else 0.0
                pen = np.maximum(fh + pr - pos[:n,:,dyn,1], 0.0)
                floor_pen = [float(pen[i].max()) for i in range(n)]
            karr = None
            for key in ['k_values','k_triplet','k_pair','k','E']:
                if key in z:
                    karr = np.asarray(z[key])[:n]
                    break
            rows.append({
                'file': os.path.basename(f), 'episodes': n_ep, 'T': T, 'transitions': T - 1, 'N': N, 'n_objects': n_objects,
                'vel_err_using_vnext': vel_err,
                'pair_contact_rate': float(np.mean(pair_contacts)) if pair_contacts else 0.0,
                'pair_max_pen_mean': float(np.mean(pair_pens)) if pair_pens else 0.0,
                'pair_max_pen_max': float(np.max(pair_pens)) if pair_pens else 0.0,
                'floor_max_pen_mean': float(np.mean(floor_pen)) if floor_pen else 0.0,
                'k_sample': karr[:min(5, len(karr))].tolist() if karr is not None else None,
            })
    for r in rows:
        print('\n' + '='*80)
        for k,v in r.items():
            print(f'{k}: {v}')

if __name__ == '__main__':
    main()
