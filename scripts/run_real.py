#!/usr/bin/env python
"""Run a trained SO-101 policy in a closed real-time control loop.

The SAME loop drives either a MuJoCo digital twin (`--robot sim`, no hardware)
or the physical arm through LeRobot (`--robot real`). Each control step:

    1. read joint angles + a camera frame from the robot
    2. build the trained observation (frame stack + proprio + grasp-site FK)
    3. run the policy -> 6-d action in [-1,1]
    4. decode the action to absolute joint targets (rad) and command the robot
    5. sleep to hold the control period (ctrl_dt, ~50 Hz)

Start on the loopback to confirm the plumbing and watch the policy attempt the
task, THEN move to hardware:

    # hardware-free digital twin (renders overhead cam at training resolution)
    MUJOCO_GL=osmesa python scripts/run_real.py --robot sim \
        --ckpt checkpoints/so101_pick_place_vision_ppo_step2112.msgpack \
        --episodes 3 --save-video /tmp/loopback.mp4

    # physical SO-101: joints via LeRobot (--port), frames via USB cam (--camera)
    # slew-rate limited; keep a hand on the e-stop
    python scripts/run_real.py --robot real --port /dev/ttyACM0 \
        --ckpt <ckpt> --camera /dev/video2 --max-step-rad 0.15
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_rl import checkpoint as ckpt_io
from vision_rl.deploy import (
    SO101Bridge, RealPolicy, SimRobot, LeRobotSO101, OpenCVCamera)


def _slew(target, prev, max_step):
    """Clamp per-joint change so the arm can't jump violently in one step."""
    if not max_step or max_step <= 0:
        return np.asarray(target, dtype=np.float64)
    return np.clip(target, prev - max_step, prev + max_step)


def _show_frame(rgb, win="model input (RGB, as seen by policy)"):
    """cv2.imshow the exact uint8 frame passed to bridge.reset/observe.

    Upscaled with INTER_NEAREST (same as scripts/camera_84x84.py) so the tiny
    training-res frame is actually visible. Returns True if 'q' was pressed.
    """
    import cv2
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    scale = max(1, 480 // bgr.shape[0])
    preview = cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale),
                          interpolation=cv2.INTER_NEAREST)
    cv2.imshow(win, preview)
    return cv2.waitKey(1) & 0xFF == ord("q")


def _move_to(robot, bridge, goal, n=40, dt=0.02, max_step=0.05):
    """Slowly slew the (real) arm to a goal pose before/after a run."""
    cur = np.asarray(robot.read_joints(), dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    for _ in range(n):
        cur = _slew(goal, cur, max_step)
        robot.send_targets(cur)
        time.sleep(dt)
        if np.max(np.abs(goal - cur)) < 1e-3:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained policy .msgpack")
    ap.add_argument("--robot", choices=["sim", "real"], default="sim")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realtime", dest="realtime", action="store_true", default=None,
                    help="hold real-time control period (default: on for real)")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false")
    ap.add_argument("--max-step-rad", type=float, default=None,
                    help="per-step joint slew limit (default: 0.15 real, off sim)")
    ap.add_argument("--save-video", type=str, default=None,
                    help="write the camera stream to this mp4/gif (needs imageio)")
    # real-hardware options
    ap.add_argument("--port", type=str, default=None, help="servo serial port (real)")
    ap.add_argument("--robot-id", type=str, default="so101")
    ap.add_argument("--camera", type=str, default="0",
                    help="USB camera index or /dev/videoN (real); frames -> 84x84 RGB")
    ap.add_argument("--grip-flip", action="store_true",
                    help="flip gripper direction if your calibration has 0=open")
    ap.add_argument("--show-camera", action="store_true",
                    help="cv2.imshow the exact frame fed to the model each step "
                         "(press 'q' in the window to stop)")
    args = ap.parse_args()

    # Config drives shapes/units; it travels inside the checkpoint.
    cfg = ckpt_io.load_config(args.ckpt)
    ep_len = int(cfg.so101.episode_length)

    realtime = args.realtime if args.realtime is not None else (args.robot == "real")
    max_step = args.max_step_rad
    if max_step is None:
        max_step = 0.15 if args.robot == "real" else 0.0

    bridge = SO101Bridge(cfg)
    policy = RealPolicy(args.ckpt, bridge.action_dim, bridge.proprio_dim)
    print(f"[run] task={cfg.task} res={cfg.render.width} cam={cfg.render.camera} "
          f"frame_stack={bridge.frame_stack} @ {policy.step:,} train-steps")

    if args.robot == "sim":
        robot = SimRobot(cfg, seed=args.seed)
    else:
        if not args.port:
            ap.error("--robot real requires --port")
        camera = OpenCVCamera(args.camera, size=cfg.render.width)
        robot = LeRobotSO101.connect(
            port=args.port, camera=camera, grip_range=bridge.gripper_ctrl_range,
            robot_id=args.robot_id, grip_flip=args.grip_flip)
        print("[run] slewing to home before start ...")
        _move_to(robot, bridge, bridge.home_targets())

    frames = []
    dt = bridge.ctrl_dt
    try:
        for ep in range(args.episodes):
            qpos = robot.reset()
            rgb = robot.read_camera()
            if args.show_camera and _show_frame(rgb):
                raise KeyboardInterrupt
            obs = bridge.reset(qpos, rgb)
            commanded = bridge.home_targets()

            loop_t0 = time.perf_counter()
            for t in range(ep_len):
                step_t0 = time.perf_counter()

                action = policy.act(obs)
                raw = bridge.action_to_targets(action)
                commanded = _slew(raw, commanded, max_step)
                robot.send_targets(commanded)

                qpos = robot.read_joints()
                rgb = robot.read_camera()
                if args.save_video:
                    frames.append(np.asarray(rgb))
                if args.show_camera and _show_frame(rgb):
                    raise KeyboardInterrupt
                obs = bridge.observe(qpos, rgb)

                if realtime:
                    lag = dt - (time.perf_counter() - step_t0)
                    if lag > 0:
                        time.sleep(lag)

                if t % 25 == 0:
                    gz = bridge.grasp_xyz(qpos)[2]
                    print(f"  ep{ep} t{t:3d}  grip={action[-1]:+.2f}  "
                          f"grasp_z={gz:.3f}m  |a|={np.abs(action[:5]).mean():.2f}")

            rate = ep_len / (time.perf_counter() - loop_t0)
            print(f"[run] episode {ep} done ({rate:.1f} Hz effective)")
    except KeyboardInterrupt:
        print("\n[run] interrupted -- stopping.")
    finally:
        if args.robot == "real":
            print("[run] slewing back to home ...")
            try:
                _move_to(robot, bridge, bridge.home_targets())
            except Exception:
                pass
        robot.close()
        if args.show_camera:
            import cv2
            cv2.destroyAllWindows()

    if args.save_video and frames:
        _write_video(args.save_video, frames)
        print(f"[run] wrote {len(frames)} frames -> {args.save_video}")


def _write_video(path: str, frames):
    import imageio
    if path.endswith(".gif"):
        imageio.mimsave(path, frames, fps=30)
    else:
        imageio.mimsave(path, frames, fps=30, macro_block_size=1)


if __name__ == "__main__":
    main()
