#!/usr/bin/env python
"""Evaluate a trained policy quantitatively.

Runs the *deterministic* policy (action = distribution mode) in the vectorized
env for a number of steps and reports success rate / reward / distance. Works
with any renderer backend (cpu / warp). For a visual rollout use view_env.py.

Examples:
    python scripts/eval.py --task so101_pick_place --backend warp \
        --ckpt checkpoints/so101_pick_place_vision_ppo_it500.msgpack --num-envs 64
    python scripts/eval.py --task franka_reach --backend cpu \
        --ckpt checkpoints/franka_reach_vision_ppo_it500.msgpack
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import serialization

from vision_rl.config import Config, so101_config
from vision_rl.envs import VisionVecEnv
from vision_rl.train import _build_network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["franka_reach", "so101_pick_place"],
                    default="so101_pick_place")
    ap.add_argument("--backend", choices=["warp", "cpu", "madrona", "auto"],
                    default="cpu")
    ap.add_argument("--ckpt", type=str, required=True, help="msgpack params file")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=4,
                    help="episodes-worth of steps to evaluate per env")
    args = ap.parse_args()

    cfg = so101_config() if args.task == "so101_pick_place" else Config()
    cfg.render.backend = args.backend
    cfg.ppo.num_envs = args.num_envs

    env = VisionVecEnv(cfg)
    net = _build_network(cfg, env.action_size)
    params = net.init(jax.random.PRNGKey(0), env.sample_obs())
    with open(args.ckpt, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    print(f"[eval] task={args.task} backend={env.renderer.backend} "
          f"num_envs={args.num_envs} ckpt={os.path.basename(args.ckpt)}")

    ep_len = (cfg.so101.episode_length if args.task == "so101_pick_place"
              else cfg.env.episode_length)
    n_steps = args.episodes * ep_len

    @jax.jit
    def rollout(vstate, params):
        def _step(vstate, _):
            pi, _ = net.apply(params, vstate.obs)
            vstate = env.step(vstate, pi.mode())        # deterministic
            m = vstate.env_state.metrics
            return vstate, {"reward": vstate.env_state.reward,
                            "success": m["success"], "dist": m["dist"],
                            "done": vstate.env_state.done}
        vstate, traj = jax.lax.scan(_step, vstate, None, length=n_steps)
        return traj

    vstate = env.reset(jax.random.PRNGKey(1))
    traj = rollout(vstate, params)
    traj = jax.tree_util.tree_map(np.asarray, traj)

    # Per-episode success: did success fire at any point within each episode?
    # Episodes are fixed length and synchronized, so reshape [E, ep_len, B].
    succ = traj["success"].reshape(args.episodes, ep_len, args.num_envs)
    ep_success = (succ.max(axis=1) > 0.5).mean()        # frac of episodes solved

    print("\n================ EVAL RESULTS ================")
    print(f"episodes evaluated : {args.episodes * args.num_envs}")
    print(f"success rate       : {ep_success:.3f}   "
          f"(fraction of episodes that reached the goal)")
    print(f"mean step reward   : {traj['reward'].mean():+.3f}")
    print(f"mean distance      : {traj['dist'].mean():.3f} m")
    print("==============================================")


if __name__ == "__main__":
    main()
