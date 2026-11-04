#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def discover(spec: str) -> List[str]:
    if os.path.isdir(spec): return sorted(glob.glob(os.path.join(spec, '*.npz')))
    if any(c in spec for c in '*?[]'): return sorted(glob.glob(spec))
    return [spec]


def locate_episode(files: List[str], global_ep: int):
    g = int(global_ep)
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            n = z['pos'].shape[0]
        if g < n: return f, g
        g -= n
    raise IndexError(global_ep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--show_floor', action='store_true')
    ap.add_argument('--show_force', action='store_true')
    ap.add_argument('--trail', action='store_true')
    ap.add_argument('--elev', type=float, default=18)
    ap.add_argument('--azim', type=float, default=-55)
    args = ap.parse_args()
    files = discover(args.data)
    f, ep = locate_episode(files, args.episode)
    with np.load(f, allow_pickle=True) as z:
        pos = np.asarray(z['pos'][ep], dtype=np.float32)
        types = np.asarray(z['types'], dtype=np.int64)
        oid = np.asarray(z['object_id'], dtype=np.int64) if 'object_id' in z else np.zeros(pos.shape[1], dtype=np.int64)
        k = None
        for key in ['k_values','k_triplet','k_pair','k','E']:
            if key in z:
                k = np.asarray(z[key][ep]).reshape(-1)
                break
        extra_mask = np.asarray(z['extra_mask'][ep], dtype=np.float32) if 'extra_mask' in z else np.zeros(pos.shape[:2], dtype=np.float32)
        extra_force = np.asarray(z['extra_force'][ep], dtype=np.float32) if 'extra_force' in z else np.zeros_like(pos)
    T,N,_ = pos.shape
    dyn = types == 0
    env = types != 0
    n_objects = int(max(1, oid[oid>=0].max()+1 if np.any(oid>=0) else 1))
    lo = pos[:, dyn].reshape(-1,3).min(axis=0); hi = pos[:, dyn].reshape(-1,3).max(axis=0)
    if env.any() and args.show_floor:
        lo = np.minimum(lo, pos[:,env].reshape(-1,3).min(axis=0)); hi = np.maximum(hi, pos[:,env].reshape(-1,3).max(axis=0))
    center = 0.5*(lo+hi); rad = float(np.max(hi-lo))*0.65 + 1e-6
    fig = plt.figure(figsize=(7,6)); ax = fig.add_subplot(111, projection='3d')
    frames = list(range(0,T,max(1,args.stride)))
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink']
    def update(t):
        ax.cla(); ax.view_init(elev=args.elev, azim=args.azim)
        title = f'ep={args.episode} frame={t}'
        if k is not None: title += ' k=' + ','.join(f'{x:g}' for x in k)
        ax.set_title(title)
        if args.show_floor and env.any():
            p = pos[t, env]
            ax.scatter(p[:,0],p[:,2],p[:,1],s=1,c='lightgray',alpha=0.25)
        for o in range(n_objects):
            idx = (types==0)&(oid==o)
            p = pos[t,idx]
            if p.size == 0: continue
            ax.scatter(p[:,0],p[:,2],p[:,1],s=8,c=colors[o%len(colors)],label=f'obj{o}')
            if args.trail:
                com = pos[:t+1,idx].mean(axis=1)
                ax.plot(com[:,0],com[:,2],com[:,1],c=colors[o%len(colors)],linewidth=1)
            if args.show_force and extra_mask[t,idx].sum() > 0:
                pf = p[extra_mask[t,idx] > 0]
                ax.scatter(pf[:,0],pf[:,2],pf[:,1],s=25,c='black')
                ff = extra_force[t,idx][extra_mask[t,idx] > 0]
                if ff.size:
                    c0 = pf.mean(axis=0); f0 = ff.mean(axis=0)
                    nrm = np.linalg.norm(f0)
                    if nrm > 1e-8:
                        f0 = f0/nrm*0.25
                        ax.quiver(c0[0],c0[2],c0[1],f0[0],f0[2],f0[1],color='black')
        ax.set_xlim(center[0]-rad, center[0]+rad); ax.set_ylim(center[2]-rad, center[2]+rad); ax.set_zlim(center[1]-rad, center[1]+rad)
        ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlabel('y')
        ax.legend(loc='upper right')
    ani = FuncAnimation(fig, update, frames=frames, interval=1000/max(1,args.fps))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    ani.save(args.out, writer=PillowWriter(fps=args.fps))
    print('[saved]', args.out)

if __name__ == '__main__':
    main()
