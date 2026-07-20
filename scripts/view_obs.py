#!/usr/bin/env python
"""Save the camera observation — the exact pixels the CNN sees — as PNG images.

Vision RL debugging: this renders through the *same* pipeline the policy uses
(env + renderer + frame-stack), grabs `obs["pixels"]`, and writes it out so you
can inspect the RGB going into the network. Works with --backend cpu or warp.

Writes two images:
  <out>_stack.png  : world-0's frame stack (the `frame_stack` frames side by side)
  <out>_grid.png   : the most-recent frame across many envs (variety the CNN sees)

Examples:
  python scripts/view_obs.py --task so101_pick_place --backend warp --steps 20
  python scripts/view_obs.py --task so101_pick_place --backend cpu --policy checkpoints/xxx.msgpack
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from PIL import Image

from vision_rl import checkpoint as ckpt_io
from vision_rl.config import Config, so101_config
from vision_rl.envs import VisionVecEnv


def _upscale(img, f):
    """Nearest-neighbour upscale so tiny 64x64 frames are actually viewable."""
    return np.repeat(np.repeat(img, f, axis=0), f, axis=1)


def _grid(frames, f, cols=None):
    n = len(frames)
    cols = cols or int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    h, w, _ = frames[0].shape
    canvas = np.zeros((rows * h * f + (rows - 1) * 4,
                       cols * w * f + (cols - 1) * 4, 3), np.uint8)
    for i, fr in enumerate(frames):
        r, c = divmod(i, cols)
        up = _upscale(fr, f)
        y, x = r * (h * f + 4), c * (w * f + 4)
        canvas[y:y + h * f, x:x + w * f] = up
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["franka_reach", "so101_pick_place"],
                    default="so101_pick_place")
    ap.add_argument("--backend", choices=["warp", "cpu"], default="cpu")
    ap.add_argument("--policy", type=str, default=None,
                    help="optional checkpoint: roll out the trained policy instead of random")
    ap.add_argument("--steps", type=int, default=15, help="env steps before grabbing obs")
    ap.add_argument("--envs", type=int, default=16, help="worlds in the grid image")
    ap.add_argument("--res", type=int, default=None,
                    help="render resolution (square), e.g. 224 or 512 (default: config 64)")
    ap.add_argument("--scale", type=int, default=None,
                    help="upscale factor for viewing (default 4 at 64px, 1 at high res)")
    ap.add_argument("--out", type=str, default="obs")
    args = ap.parse_args()

    # With --policy the checkpoint defines the config; without it, --task picks a
    # default one (this script also runs standalone to preview random rollouts).
    ck = ckpt_io.load(args.policy) if args.policy else None
    cfg = ck.config if ck else (
        so101_config() if args.task == "so101_pick_place" else Config())
    cfg.render.backend = args.backend
    cfg.ppo.num_envs = max(args.envs, 1)
    if args.res:
        if args.policy:
            print("WARNING: --policy expects the resolution it was trained at "
                  f"({cfg.render.width}); --res {args.res} will fail to load.")
        cfg.render.width = cfg.render.height = args.res
    # sensible default upscale: big for tiny frames, none for high-res
    args.scale = args.scale or (1 if cfg.render.width >= 160 else 4)

    env = VisionVecEnv(cfg)
    act = None
    if ck:
        from vision_rl.train import _build_network
        net = _build_network(cfg, env.action_size)
        params = ck.restore_params(net.init(jax.random.PRNGKey(0), env.sample_obs()))
        act = jax.jit(lambda o: net.apply(params, o)[0].mode())

    rng = jax.random.PRNGKey(0)
    vs = env.reset(rng)
    for _ in range(args.steps):
        rng, k = jax.random.split(rng)
        if act is not None:
            action = act(vs.obs)
        else:
            action = jax.random.uniform(k, (cfg.ppo.num_envs, env.action_size),
                                        minval=-1.0, maxval=1.0)
        vs = env.step(vs, action)

    pix = np.asarray(vs.obs["pixels"])          # [B, H, W, 3*frame_stack] uint8
    k = cfg.encoder.frame_stack
    print(f"obs pixels: shape={pix.shape} dtype={pix.dtype} "
          f"min={pix.min()} max={pix.max()} mean={pix.mean():.1f}")

    # world-0 frame stack, oldest -> newest
    w0 = pix[0]
    frames = [w0[..., i * 3:(i + 1) * 3] for i in range(k)]
    strip = np.concatenate([_upscale(f, args.scale) for f in frames], axis=1)
    p1 = f"{args.out}_stack.png"
    Image.fromarray(strip).save(p1)

    # newest frame across envs
    cur = [pix[i, :, :, -3:] for i in range(min(args.envs, pix.shape[0]))]
    p2 = f"{args.out}_grid.png"
    Image.fromarray(_grid(cur, args.scale)).save(p2)

    print(f"saved {p1}  (world-0's {k} stacked frames, oldest->newest)")
    print(f"saved {p2}  ({len(cur)} worlds, most-recent frame)")


if __name__ == "__main__":
    main()
