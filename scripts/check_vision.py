#!/usr/bin/env python
"""Diagnose whether a trained policy actually USES the camera.

The cube/target positions are only observable through pixels (they are not in
proprio). So if the policy is vision-conditioned, its action must change when
the cube moves. If the action is nearly constant across very different cube
positions, the policy has collapsed to a blind, state-independent behaviour.

Reports:
  * action std across worlds  -> ~0 means a constant (blind) action
  * |corr(action, cube_x/y)|  -> ~0 means the action ignores where the cube is

Example:
  python scripts/check_vision.py --task so101_pick_place --backend warp \
      --ckpt checkpoints/so101_pick_place_vision_ppo_step44000000.msgpack
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
from flax import serialization

from vision_rl.config import Config, so101_config
from vision_rl.envs import VisionVecEnv
from vision_rl.train import _build_network, ckpt_render_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["franka_reach", "so101_pick_place"],
                    default="so101_pick_place")
    ap.add_argument("--backend", choices=["warp", "cpu"], default="cpu")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--envs", type=int, default=64, help="worlds (each a different cube pos)")
    ap.add_argument("--res", type=int, default=None,
                    help="override the resolution inferred from the checkpoint")
    args = ap.parse_args()

    cfg = so101_config() if args.task == "so101_pick_place" else Config()
    cfg.render.backend = args.backend
    cfg.ppo.num_envs = args.envs

    res = args.res or ckpt_render_res(args.ckpt, cfg)
    if res and res != cfg.render.width:
        print(f"[ckpt] trained at {res}x{res}; config default is "
              f"{cfg.render.width}x{cfg.render.height} -> rendering at {res}")
        cfg.render.width = cfg.render.height = res

    env = VisionVecEnv(cfg)
    net = _build_network(cfg, env.action_size)
    params = net.init(jax.random.PRNGKey(0), env.sample_obs())
    with open(args.ckpt, "rb") as f:
        params = serialization.from_bytes(params, f.read())

    # Each world resets with a DIFFERENT random cube/target; arm starts ~same pose.
    vs = env.reset(jax.random.PRNGKey(7))
    pi, _ = net.apply(params, vs.obs)
    actions = np.asarray(pi.mode())                    # [B, action_dim]

    base = env.env
    qpos = np.asarray(vs.env_state.data.qpos)          # [B, nq]
    cube_xy = qpos[:, base._cube_qadr:base._cube_qadr + 2]   # [B, 2]

    print(f"\ncheckpoint: {os.path.basename(args.ckpt)}   worlds: {args.envs}")
    print(f"cube x spread: {cube_xy[:,0].std():.4f} m   y spread: {cube_xy[:,1].std():.4f} m")
    print("\n--- action variability across very different cube positions ---")
    per_dim_std = actions.std(axis=0)
    for i, s in enumerate(per_dim_std):
        name = f"joint{i}" if i < env.action_size - 1 else "gripper"
        print(f"  action[{i}] ({name:<7}) std = {s:.4f}")
    print(f"  MEAN action std = {per_dim_std.mean():.4f}")

    print("\n--- does the action depend on WHERE the cube is? ---")
    for i in range(actions.shape[1]):
        cx = abs(np.corrcoef(actions[:, i], cube_xy[:, 0])[0, 1])
        cy = abs(np.corrcoef(actions[:, i], cube_xy[:, 1])[0, 1])
        print(f"  action[{i}]  |corr with cube_x| = {cx:.3f}   |corr with cube_y| = {cy:.3f}")

    max_corr = max(
        max(abs(np.corrcoef(actions[:, i], cube_xy[:, j])[0, 1]) for j in (0, 1))
        for i in range(actions.shape[1]))
    print("\n================= VERDICT =================")
    if per_dim_std.mean() < 0.02 or max_corr < 0.15:
        print("BLIND POLICY: action barely changes with the cube position.")
        print("  -> the CNN is being ignored; it learned one fixed pose.")
    else:
        print("VISION-CONDITIONED: action tracks the cube position.")
    print(f"  mean action std = {per_dim_std.mean():.4f}, max |corr| = {max_corr:.3f}")
    print("==========================================")


if __name__ == "__main__":
    main()
