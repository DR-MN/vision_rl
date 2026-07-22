#!/usr/bin/env python
"""Prove the deploy bridge reproduces the training env's action decode + proprio.

The whole point of vision_rl/deploy is that a policy deployed on hardware sees
exactly what it saw in training and its actions are commanded exactly as trained.
This asserts that numerically against the real SO101PickPlaceEnv:

    * action_to_targets(a)  == the ctrl the env commands for the same action
    * grasp_xyz(qpos)       == the env's proprio gripper-site position (FK)
    * proprio joint block    == the env's qpos block

Run (CPU is fine and fastest to compile):
    JAX_PLATFORMS=cpu MUJOCO_GL=osmesa python scripts/verify_deploy.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from vision_rl.config import so101_config
from vision_rl.envs import make_env
from vision_rl.deploy.bridge import SO101Bridge


def main():
    cfg = so101_config()
    cfg.render.backend = "cpu"          # -> physics impl='jax'
    cfg.ppo.num_envs = 1

    env = make_env(cfg)
    bridge = SO101Bridge(cfg)

    vreset = jax.jit(jax.vmap(env.reset))
    step_ctrl = jax.jit(lambda st, a: jax.vmap(env.step)(st, a).data.ctrl)
    state = vreset(jax.random.split(jax.random.PRNGKey(3), 1))

    # 1) action decode: bridge vs the ctrl the env actually commands.
    rng = np.random.default_rng(0)
    max_ctrl_err = 0.0
    for _ in range(20):
        a = rng.uniform(-1.5, 1.5, size=bridge.action_dim)   # incl. out-of-range
        env_ctrl = np.asarray(step_ctrl(state, jnp.asarray(a)[None])[0])
        max_ctrl_err = max(max_ctrl_err,
                           np.max(np.abs(env_ctrl - bridge.action_to_targets(a))))

    # 2) proprio + grasp-site forward kinematics.
    env_proprio = np.asarray(state.obs[0])
    qpos6 = np.asarray(state.data.qpos[0, :6])
    grasp_err = np.max(np.abs(env_proprio[-3:] - bridge.grasp_xyz(qpos6)))
    qpos_err = np.max(np.abs(env_proprio[:6] - qpos6))

    print(f"action decode  max|env-bridge ctrl| = {max_ctrl_err:.2e} rad")
    print(f"grasp-site FK  max|env-bridge xyz|  = {grasp_err:.2e} m")
    print(f"proprio qpos   max|env-bridge|      = {qpos_err:.2e} rad")
    ok = max_ctrl_err < 1e-5 and grasp_err < 1e-4 and qpos_err < 1e-6
    print("RESULT:", "PASS -- deploy matches training" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
