#!/usr/bin/env python
"""Interactive sim<->real calibration check: a MuJoCo "ghost" arm that the real
SO-101 is commanded to mirror, so a joint-zero/sign/range mismatch is visible
directly instead of inferred from a crash.

scripts/verify_deploy.py only proves the bridge's own math is self-consistent
(action decode + FK match the training env) -- it says nothing about whether
the sim's joint convention matches the *real* servo calibration. This tool
checks that correspondence directly and interactively:

    1. Spawns a MuJoCo viewer showing the SO-101 model used in training.
    2. If --port is given, connects to the real arm (read-only until you
       press ENTER -- nothing is commanded on connect).
    3. ENTER -> the sim arm slews to the trained HOME pose over a few seconds
       (small per-tick step, never a jump), and the real arm is commanded to
       follow the same trajectory. Watch the physical arm against the GUI:
       if it doesn't track the same pose, or dives toward the table, you have
       a calibration mismatch -- Ctrl+C immediately.
    4. Keys 1-6 select a joint (shoulder_pan..gripper, see JOINT_NAMES);
       +/- nudges ONLY that joint by --jog-step rad (also slewed). This
       isolates which joint is miscalibrated instead of guessing from a
       combined motion.
    5. 'q' / ESC / Ctrl+C quits. The arm is NOT auto-moved anywhere on exit
       (home itself may be unverified) -- it just holds its last commanded
       position; power it down or move it by hand once you trust it.

SAFETY: keep a hand near the physical e-stop / power switch for the first
ENTER press on real hardware -- that is exactly the untrusted transition this
tool exists to de-risk. Start with the default --max-step-rad (conservative).

Usage:
    # sim-only dry run, no hardware -- just look at the GUI and the numbers
    python scripts/check_calibration.py --task so101_pick_place

    # mirror onto the real arm
    python scripts/check_calibration.py --task so101_pick_place \
        --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_rl.config import so101_config, so101_pick_config
from vision_rl.deploy import SO101Bridge, LeRobotSO101

MISMATCH_DEG = 10.0    # |real - commanded| beyond this is flagged, not just tracking lag


def _slew(target, prev, max_step):
    return np.clip(target, prev - max_step, prev + max_step)


def _fmt_row(name, cmd_rad, real_rad, selected):
    marker = ">" if selected else " "
    cmd_deg = math.degrees(cmd_rad)
    if real_rad is None:
        return f"{marker} {name:14s} cmd={cmd_deg:7.2f}deg"
    real_deg = math.degrees(real_rad)
    delta = real_deg - cmd_deg
    flag = "  !! MISMATCH" if abs(delta) > MISMATCH_DEG else ""
    return (f"{marker} {name:14s} cmd={cmd_deg:7.2f}deg  real={real_deg:7.2f}deg  "
            f"d={delta:+6.2f}deg{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["so101_pick_place", "so101_pick"],
                     default="so101_pick")
    ap.add_argument("--port", type=str, default=None,
                     help="real servo serial port; omit for a sim-only dry run")
    ap.add_argument("--robot-id", type=str, default="so101")
    ap.add_argument("--grip-flip", action="store_true")
    ap.add_argument("--max-step-rad", type=float, default=0.03,
                     help="per-tick slew cap (rad) -- how fast commanded targets "
                          "are allowed to move; keep small until calibration is verified")
    ap.add_argument("--jog-step", type=float, default=0.05,
                     help="rad added/removed from the selected joint per +/- press")
    ap.add_argument("--hz", type=float, default=25.0, help="control/print loop rate")
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        raise SystemExit("scripts/check_calibration.py needs opencv-python for "
                          "keyboard input: pip install opencv-python")

    cfg = so101_config() if args.task == "so101_pick_place" else so101_pick_config()
    bridge = SO101Bridge(cfg)
    names = bridge.joint_names
    n = len(names)
    limits = bridge.joint_limits_rad
    home = bridge.home_targets()

    print(f"[calib] task={args.task}  sim joint ranges + home:")
    for nm in names:
        lo, hi = limits[nm]
        ho = float(home[names.index(nm)])
        print(f"  {nm:14s} [{math.degrees(lo):7.2f}, {math.degrees(hi):7.2f}] deg  "
              f"home={math.degrees(ho):7.2f} deg")

    model = bridge.model
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]          # cube/target etc. sit at their home pose;
                                               # only the 6 robot dofs are driven below

    robot = None
    if args.port:
        robot = LeRobotSO101.connect(
            port=args.port, camera=None, grip_range=bridge.gripper_ctrl_range,
            robot_id=args.robot_id, grip_flip=args.grip_flip)
        commanded = robot.read_joints()
        print("[calib] connected -- mirroring the arm's CURRENT pose into the GUI "
              "(nothing commanded yet). Press ENTER in the control window to slew home.")
    else:
        commanded = np.zeros(n, dtype=np.float64)
        print("[calib] --port not given: sim-only dry run, no hardware commanded.")

    current_target = commanded.copy()
    selected = 0

    data.qpos[:n] = commanded
    mujoco.mj_forward(model, data)
    viewer = mujoco.viewer.launch_passive(model, data)

    ctrl_win = "check_calibration -- ENTER=home 1-6=select joint +/-=jog q=quit"
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ctrl_win, 640, 200)

    dt = 1.0 / args.hz
    last_print = 0.0
    try:
        running = True
        while running:
            t0 = time.perf_counter()

            panel = np.full((200, 640, 3), 30, dtype=np.uint8)
            cv2.putText(panel, "ENTER: slew to HOME", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(panel, "1-6: select joint (see terminal)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(panel, f"+/-: jog selected joint (+-{args.jog_step:.3f} rad)",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(panel, f"selected: {names[selected]}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(panel, "q / ESC: quit (holds last position)", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(ctrl_win, panel)

            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):
                current_target = home.copy()
                print("[calib] -> slewing to HOME")
            elif ord("1") <= key <= ord(str(min(6, n))):
                selected = key - ord("1")
                print(f"[calib] selected joint: {names[selected]}")
            elif key in (ord("+"), ord("=")):
                lo, hi = limits[names[selected]]
                current_target[selected] = float(np.clip(
                    current_target[selected] + args.jog_step, lo, hi))
                print(f"[calib] jog {names[selected]} -> "
                      f"{math.degrees(current_target[selected]):.2f} deg")
            elif key in (ord("-"), ord("_")):
                lo, hi = limits[names[selected]]
                current_target[selected] = float(np.clip(
                    current_target[selected] - args.jog_step, lo, hi))
                print(f"[calib] jog {names[selected]} -> "
                      f"{math.degrees(current_target[selected]):.2f} deg")
            elif key in (ord("q"), 27):
                running = False

            commanded = _slew(current_target, commanded, args.max_step_rad)
            data.qpos[:n] = commanded
            mujoco.mj_forward(model, data)
            if viewer.is_running():
                viewer.sync()
            else:
                running = False

            real = None
            if robot is not None:
                robot.send_targets(commanded)
                real = robot.read_joints()

            now = time.perf_counter()
            if now - last_print > 0.3:
                last_print = now
                lines = [_fmt_row(nm, commanded[i], None if real is None else real[i],
                                   i == selected) for i, nm in enumerate(names)]
                print("\n".join(lines) + "\n")

            lag = dt - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)
    except KeyboardInterrupt:
        print("\n[calib] interrupted.")
    finally:
        print("[calib] holding last commanded position; not auto-moving anywhere.")
        if robot is not None:
            robot.close()
        try:
            viewer.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
