#!/usr/bin/env python
"""Faithful SO-101 viewer — deterministic (deployment) policy.

Supports BOTH SO-101 tasks (picked automatically from the checkpoint's
cfg.task): so101_pick_place (pick + carry to a target pad) and so101_pick
(pick + lift above pick_height, no place stage).

Why this exists (vs view_env.py):
    view_env.py re-implements the dynamics with plain `mj_step` and can drift
    from the real training env. This script instead drives the ACTUAL training
    env (SO101PickPlaceEnv or SO101PickEnv: menagerie SO-101, real physical
    grasping, staged reward, identical dynamics) and mirrors its state into a
    passive MuJoCo window, so what you see matches training exactly.

Deterministic only: a deployed / real-world policy uses the mean action
(`pi.mode()`), not the exploration noise used while training, so that is what we
show — no stochastic flag.

    MUJOCO_GL=glfw python scripts/view_env2.py \
        --ckpt checkpoints/so101_pick_place_vision_ppo_step49804800.msgpack

    MUJOCO_GL=glfw python scripts/view_env2.py \
        --ckpt checkpoints/so101_pick_vision_ppo_20260723-105813/so101_pick_vision_ppo_step4992.msgpack

    # verify a checkpoint without a display (prints grasp/lift/success stats):
    MUJOCO_GL=osmesa python scripts/view_env2.py --ckpt <ckpt> --headless 250
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
import jax
import jax.numpy as jnp
from vision_rl import checkpoint as ckpt_io
from vision_rl.envs.so101_pick_place import SO101PickPlaceEnv
from vision_rl.envs.so101_pick import SO101PickEnv
from vision_rl.train import _build_network

_ENV_BY_TASK = {
    "so101_pick_place": lambda cfg: SO101PickPlaceEnv(cfg.so101, impl="jax"),
    "so101_pick": lambda cfg: SO101PickEnv(cfg.pick, impl="jax"),
}
# What the final "success" metric means per task, for the headless stats label.
_SUCCESS_LABEL = {"so101_pick_place": "placed", "so101_pick": "picked"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained params .msgpack")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", type=int, default=0, metavar="STEPS",
                    help="run without a window for STEPS steps, print pick stats")
    args = ap.parse_args()

    ck = ckpt_io.load(args.ckpt)
    cfg = ck.config                      # resolution, reward shaping, action scale, task
    res = cfg.render.width
    k = cfg.encoder.frame_stack

    if cfg.task not in _ENV_BY_TASK:
        raise ValueError(f"view_env2.py supports {list(_ENV_BY_TASK)}, got {cfg.task!r}")

    # The real training env (CPU MJX): physical grasping + exact staged dynamics.
    env = _ENV_BY_TASK[cfg.task](cfg)

    # Policy built exactly as in training (tanh mean, LayerNorm, log_std clamp).
    net = _build_network(cfg, env.action_size)
    dummy = {"pixels": jnp.zeros((1, res, res, k * 3), jnp.uint8),
             "proprio": jnp.zeros((1, env.proprio_size), jnp.float32)}
    params = ck.restore_params(net.init(jax.random.PRNGKey(0), dummy))

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    @jax.jit
    def policy(obs):
        pi, _ = net.apply(params, obs)
        return pi.mode()[0]                       # deterministic; single env -> [6]

    # Classic model/data: renders the policy's camera *and* backs the window.
    model = mujoco.MjModel.from_xml_path(os.path.abspath(env.xml_path))
    data = mujoco.MjData(model)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.render.camera)
    renderer = mujoco.Renderer(model, res, res)

    def mirror(s):
        """Copy MJX env state into the classic MjData for render + display."""
        data.qpos[:] = np.asarray(s.data.qpos)
        data.qvel[:] = np.asarray(s.data.qvel)
        if model.nmocap:
            data.mocap_pos[:] = np.asarray(s.data.mocap_pos).reshape(model.nmocap, 3)
        mujoco.mj_forward(model, data)

    frames = {"buf": None}

    def get_obs(s):
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render().astype(np.uint8)                 # [H,W,3]
        b = frames["buf"]
        frames["buf"] = (np.tile(rgb, (1, 1, k)) if b is None
                         else np.concatenate([b[..., 3:], rgb], axis=-1))
        return {"pixels": frames["buf"][None], "proprio": np.asarray(s.obs)[None]}

    def show_camera_frame(pixels, win="model input (RGB, as seen by policy)"):
        """cv2.imshow the exact newest frame inside obs['pixels'] (mirrors
        scripts/run_real.py's _show_frame, upscaled with INTER_NEAREST so the
        tiny training-res frame is actually visible)."""
        import cv2
        rgb = np.asarray(pixels)[0, ..., -3:]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        scale = max(1, 480 // bgr.shape[0])
        preview = cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)
        cv2.imshow(win, preview)
        cv2.waitKey(1)

    ep = env.cfg.episode_length
    key = jax.random.PRNGKey(args.seed)
    s = reset(key)

    # ---------------- headless verification ----------------
    if args.headless:
        grasped = lifted = success = False
        n_ep = 0; picks = 0; lifts = 0; succ = 0
        for i in range(args.headless):
            mirror(s)
            action = np.asarray(policy(get_obs(s)))
            s = step(s, jnp.asarray(action))
            grasped |= bool(s.grasped > 0.5)
            lifted |= bool(s.metrics["cube_z"] > env.cfg.lift_thresh)
            success |= bool(s.metrics["success"] > 0.5)
            if bool(s.done) or (i + 1) % ep == 0:
                n_ep += 1; picks += grasped; lifts += lifted; succ += success
                key, sub = jax.random.split(key); s = reset(sub)
                frames["buf"] = None
                grasped = lifted = success = False
        succ_label = _SUCCESS_LABEL[cfg.task]
        print(f"\n[headless] task={cfg.task} deterministic policy over {n_ep} episode(s), res={res}")
        print(f"  grasped        : {picks}/{n_ep}")
        print(f"  lifted         : {lifts}/{n_ep}")
        print(f"  {succ_label:<15}: {succ}/{n_ep}")
        print("  (physical grasping — the gripper must really close on the cube)")
        return

    # ---------------- interactive window ----------------
    print(f"[view_env2] task={cfg.task} | deterministic (pi.mode) | res={res} | physical grasping")
    print("            press Tab / '[' / ']' to cycle cameras (incl. overhead_cam)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0
        while viewer.is_running():
            t0 = time.time()
            mirror(s)                         # show the state the policy will act on
            viewer.sync()
            obs = get_obs(s)
            show_camera_frame(obs["pixels"])
            action = np.asarray(policy(obs))
            s = step(s, jnp.asarray(action))
            t += 1
            if bool(s.done) or t >= ep:
                key, sub = jax.random.split(key)
                s = reset(sub); frames["buf"] = None; t = 0
            dt = env.cfg.ctrl_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    import cv2
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
