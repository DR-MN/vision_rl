#!/usr/bin/env python
"""Interactive MuJoCo GUI viewer for the vision-RL tasks.

Opens the native MuJoCo viewer window (needs a display, e.g. DISPLAY=:0) and
drives the *same* dynamics as the training env using classic MuJoCo on CPU.

Tasks:
    --task so101_pick_place   (default) SO-101 picks a cube and places it on the pad

Modes:
    (default)          hold the home pose -- just look around
    --random           apply random actions
    --policy CKPT      roll out a trained policy (deterministic)

Examples:
    MUJOCO_GL=glfw python scripts/view_env.py --task so101_pick_place
    MUJOCO_GL=glfw python scripts/view_env.py --task so101_pick_place --random
    MUJOCO_GL=glfw python scripts/view_env.py --policy checkpoints/....msgpack

Tip: press Tab / '[' / ']' in the window to cycle cameras (incl. overhead_cam).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer

from vision_rl.config import so101_config

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision_rl", "envs", "assets")

# Per-task viewer settings.
TASKS = {
    "so101_pick_place": dict(
        scene=os.path.join(_ASSETS, "so101_menagerie", "so101_pick_place.xml"),
        n_arm=5, gripper_action=True, site="gripperframe",
    ),
}


def _load_policy(ck, action_size, proprio_size, pixels_shape):
    import jax
    import jax.numpy as jnp

    from vision_rl.train import _build_network

    # _build_network is the same constructor training uses, so options like
    # layer_norm and the log_std clamp can't silently drift out of sync here.
    net = _build_network(ck.config, action_size)
    dummy = {"pixels": jnp.zeros((1, *pixels_shape), jnp.uint8),
             "proprio": jnp.zeros((1, proprio_size), jnp.float32)}
    params = ck.restore_params(net.init(jax.random.PRNGKey(0), dummy))

    @jax.jit
    def act(params, obs):
        pi, _ = net.apply(params, obs)
        return pi.mode()

    return lambda obs: np.asarray(act(params, obs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="so101_pick_place")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--policy", type=str, default=None)
    args = ap.parse_args()

    # With --policy the checkpoint is the source of truth for task and config;
    # without it, --task selects a default one for a random/scripted rollout.
    ck = None
    if args.policy:
        from vision_rl import checkpoint as ckpt_io
        ck = ckpt_io.load(args.policy)
        cfg = ck.config
        args.task = cfg.task            # keeps the T/ctrl_dt lookups below honest
    else:
        cfg = so101_config()

    T = TASKS[args.task]
    rng = np.random.default_rng(0)
    n_arm = T["n_arm"]

    model = mujoco.MjModel.from_xml_path(os.path.abspath(T["scene"]))
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep
    ctrl_dt = cfg.so101.ctrl_dt
    action_scale = cfg.so101.action_scale
    n_substeps = max(1, round(ctrl_dt / sim_dt))

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, T["site"])
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
    home_ctrl = model.key_ctrl[0].copy()
    home_qpos = model.key_qpos[0].copy()
    arm_low = model.actuator_ctrlrange[:n_arm, 0]
    arm_high = model.actuator_ctrlrange[:n_arm, 1]
    n_robot = n_arm + (1 if T["gripper_action"] else 0)

    def randomize():
        """Per-episode randomization matching the training env."""
        data.qpos[:] = home_qpos
        data.qpos[:n_arm] += 0.05 * rng.standard_normal(n_arm)
        data.qvel[:] = 0.0
        data.ctrl[:] = home_ctrl
        cb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        qadr = model.jnt_qposadr[model.body_jntadr[cb]]
        data.qpos[qadr:qadr+2] = rng.uniform(cfg.so101.cube_low, cfg.so101.cube_high)
        data.qpos[qadr+2] = cfg.so101.cube_z
        tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "place_target")
        data.mocap_pos[model.body_mocapid[tb], :2] = rng.uniform(
            cfg.so101.target_low, cfg.so101.target_high)
        mujoco.mj_forward(model, data)

    policy = renderer = None
    frames = None
    if args.policy:
        pixels_shape = (cfg.render.height, cfg.render.width, cfg.encoder.frame_stack * 3)
        proprio_size = n_robot * 2 + 3
        action_size = n_arm + (1 if T["gripper_action"] else 0)
        policy = _load_policy(ck, action_size, proprio_size, pixels_shape)
        renderer = mujoco.Renderer(model, cfg.render.height, cfg.render.width)

    def get_obs():
        nonlocal frames
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render().astype(np.uint8)
        k = cfg.encoder.frame_stack
        frames = (np.tile(rgb, (1, 1, k)) if frames is None
                  else np.concatenate([frames[..., 3:], rgb], axis=-1))
        proprio = np.concatenate([data.qpos[:n_robot], data.qvel[:n_robot],
                                  data.site_xpos[site]]).astype(np.float32)
        return {"pixels": frames[None], "proprio": proprio[None]}

    randomize()
    step_count = 0
    ep_len = cfg.so101.episode_length
    print(f"[viewer] task={args.task} mode=",
          "policy" if policy else ("random" if args.random else "hold-home"))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            if policy is not None:
                action = policy(get_obs())[0]
            elif args.random:
                action = rng.uniform(-1.0, 1.0, n_robot)
            else:
                action = np.zeros(n_robot)

            data.ctrl[:n_arm] = np.clip(
                home_ctrl[:n_arm] + action_scale * action[:n_arm], arm_low, arm_high)
            if T["gripper_action"]:
                glow, ghigh = model.actuator_ctrlrange[n_arm]
                g = 0.5 * (action[n_arm] + 1.0)
                data.ctrl[n_arm] = glow + g * (ghigh - glow)
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)

            step_count += 1
            if step_count >= ep_len:
                randomize(); frames = None; step_count = 0

            viewer.sync()
            dt = ctrl_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
