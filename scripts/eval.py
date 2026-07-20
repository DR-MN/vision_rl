#!/usr/bin/env python
"""Evaluate a trained policy quantitatively.

Runs the *deterministic* policy (action = distribution mode) in the vectorized
env for a number of steps and reports success rate / reward / distance. Works
with any renderer backend (cpu / warp). For a visual rollout use view_env.py.

Examples:
    python scripts/eval.py --backend warp --num-envs 64 \
        --ckpt checkpoints/so101_pick_place_vision_ppo_it500.msgpack
    python scripts/eval.py --backend cpu \
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
from vision_rl import checkpoint as ckpt_io
from vision_rl.envs import VisionVecEnv
from vision_rl.train import _build_network


def main():
    ap = argparse.ArgumentParser()
    # No --task: the checkpoint records which task it was trained on.
    ap.add_argument("--backend", choices=["warp", "cpu", "madrona", "auto"],
                    default="cpu")
    ap.add_argument("--ckpt", type=str, required=True, help="msgpack params file")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=4,
                    help="episodes-worth of steps to evaluate per env")
    ap.add_argument("--res", type=int, default=None,
                    help="override the resolution inferred from the checkpoint")
    args = ap.parse_args()

    # Every setting that shaped the policy -- task, resolution, reward shaping,
    # episode length -- comes from the checkpoint, so eval reproduces training
    # conditions exactly. Only backend and world count are ours to pick.
    ck = ckpt_io.load(args.ckpt)
    cfg = ck.config
    cfg.render.backend = args.backend
    cfg.ppo.num_envs = args.num_envs
    if args.res:
        cfg.render.width = cfg.render.height = args.res
        print(f"[eval] WARNING: overriding resolution to {args.res}x{args.res}; "
              f"the policy was trained at {ck.config.render.width}")

    env = VisionVecEnv(cfg)
    net = _build_network(cfg, env.action_size)
    params = ck.restore_params(net.init(jax.random.PRNGKey(0), env.sample_obs()))
    print(f"[eval] task={cfg.task} backend={env.renderer.backend} "
          f"num_envs={args.num_envs} res={cfg.render.width} "
          f"ckpt={os.path.basename(args.ckpt)} @ {ck.step:,} steps")

    ep_len = (cfg.so101.episode_length if cfg.task == "so101_pick_place"
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
