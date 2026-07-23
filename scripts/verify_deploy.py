#!/usr/bin/env python
"""Prove the deploy bridge reproduces the training env's action decode + proprio.

The whole point of vision_rl/deploy is that a policy deployed on hardware sees
exactly what it saw in training and its actions are commanded exactly as trained.
This asserts that numerically against the real SO101PickPlaceEnv:

    * action_to_targets(a)  == the ctrl the env commands for the same action
    * grasp_xyz(qpos)       == the env's proprio gripper-site position (FK)
    * proprio joint block    == the env's qpos block

KNOWN, ACCEPTED GAP (2026-07-23): grasp_xyz vs the env's own reported grasp
position can differ by a few mm (not more than GRASP_TOL below). This is NOT a
bug in bridge.py -- MJX's `data.site_xpos` after `mjx.step()` reflects the qpos
from BEFORE that step's integration (same convention as classic MuJoCo's
mj_step: forward-at-current-qpos, then integrate, no re-kinematics after).
Isolated single-substep test confirmed this exactly (site_xpos matches classic
FK of the PRE-step qpos, not the POST-step qpos MJX returns alongside it). It
therefore affects the training env's own reward/observation too (not just this
deploy check), in both so101_pick_place.py and so101_pick.py. Decision (user):
small relative to the 4.5cm cube and dwarfed by the vision domain-gap risk --
leave the physics alone, just don't fail this check on it.

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

# grasp-site tolerance: loose enough to absorb the known one-substep-stale
# kinematics gap (see module docstring), tight enough to still catch a real
# FK bug (e.g. a wrong site name/index, which would be off by cm, not mm).
GRASP_TOL_M = 6e-3


def main():
    cfg = so101_config()
    cfg.render.backend = "cpu"          # -> physics impl='jax'
    cfg.ppo.num_envs = 1

    env = make_env(cfg)
    bridge = SO101Bridge(cfg)

    vreset = jax.jit(jax.vmap(env.reset))
    vstep = jax.jit(jax.vmap(env.step))
    state = vreset(jax.random.split(jax.random.PRNGKey(3), 1))

    # 1) action decode: bridge vs the ctrl the env actually commands. Stepped
    # SEQUENTIALLY (not 20 independent actions from a frozen state) because
    # action_to_targets is now stateful -- it slew-limits relative to the
    # PREVIOUSLY commanded arm target, same as the env's data.ctrl carries
    # across steps. Both sides start from home (fresh reset / fresh bridge),
    # so their slew references stay in lockstep as long as we feed the same
    # action to both at each step.
    rng = np.random.default_rng(0)
    max_ctrl_err = 0.0
    for _ in range(20):
        a = rng.uniform(-1.5, 1.5, size=bridge.action_dim)   # incl. out-of-range
        bridge_ctrl = bridge.action_to_targets(a)
        state = vstep(state, jnp.asarray(a)[None])
        env_ctrl = np.asarray(state.data.ctrl[0])
        max_ctrl_err = max(max_ctrl_err, np.max(np.abs(env_ctrl - bridge_ctrl)))

    # 2) proprio + grasp-site forward kinematics.
    env_proprio = np.asarray(state.obs[0])
    qpos6 = np.asarray(state.data.qpos[0, :6])
    grasp_err = np.max(np.abs(env_proprio[-3:] - bridge.grasp_xyz(qpos6)))
    qpos_err = np.max(np.abs(env_proprio[:6] - qpos6))

    print(f"action decode  max|env-bridge ctrl| = {max_ctrl_err:.2e} rad")
    print(f"grasp-site FK  max|env-bridge xyz|  = {grasp_err:.2e} m  "
          f"(tol {GRASP_TOL_M:.0e} m -- absorbs the known 1-substep kinematics lag)")
    print(f"proprio qpos   max|env-bridge|      = {qpos_err:.2e} rad")
    ok = max_ctrl_err < 1e-5 and grasp_err < GRASP_TOL_M and qpos_err < 1e-6
    print("RESULT:", "PASS -- deploy matches training" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
