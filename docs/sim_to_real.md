# Sim-to-real: deploying the SO-101 vision policy

This explains how a policy trained in MJX is connected to the physical SO-101,
and what still has to be done to make it actually work on hardware.

## What the policy consumes and produces

The trained network ([models/networks.py](../vision_rl/models/networks.py)) is a
pure function `obs -> (action, value)`:

| | shape | meaning |
|---|---|---|
| **obs["pixels"]** | uint8 `[H, W, 3·frame_stack]` (84×84×9) | last 3 RGB frames from `overhead_cam`, channel-stacked |
| **obs["proprio"]** | float `[15]` | joint pos (6) + joint vel (6) + gripper-site xyz (3) |
| **action** | float `[6]` in `[-1,1]` | `[0:5]` residual arm-joint targets, `[5]` gripper open/close |

Control runs at **50 Hz** (`ctrl_dt = 0.02`). The arm actuators are MuJoCo
*position* actuators whose ranges equal the real Feetech STS3215 servo ranges, so
the action decode maps 1:1 onto real position commands.

## The three connections (all in `vision_rl/deploy/`)

1. **model action → real joints** — [bridge.py](../vision_rl/deploy/bridge.py)
   `SO101Bridge.action_to_targets` reproduces the env's decode byte-for-byte:
   `arm = home + 0.9·a[:5]` (clipped per joint); gripper `[-1,1] → [closed,open]`.
   The result is absolute joint targets in **radians** → send to the servos.

2. **real joints → proprio** — `SO101Bridge.observe` reads the 6 servo angles,
   finite-differences them for velocity, and computes the gripper-site xyz with
   **MuJoCo forward kinematics on the same model/site the env used** (`grasp_xyz`),
   guaranteeing the proprio vector matches training.

3. **real camera → pixels** — [camera.py](../vision_rl/deploy/camera.py)
   `OpenCVCamera` grabs USB-camera frames and returns 84×84 RGB uint8 (same as
   [scripts/camera_84x84.py](../scripts/camera_84x84.py)); `SO101Bridge` keeps a
   rolling 3-frame stack (`reset`/`observe`) — mirroring
   [vision_wrapper.py](../vision_rl/envs/vision_wrapper.py). Camera is decoupled
   from LeRobot: **joints via LeRobot, frames via OpenCV.**

Inference is [policy.py](../vision_rl/deploy/policy.py) `RealPolicy` (batch-1,
jitted, deterministic `pi.mode()`), reusing the training network builder and
checkpoint loader unchanged. Hardware sits behind
[robots.py](../vision_rl/deploy/robots.py): `SimRobot` (MuJoCo loopback) and
`LeRobotSO101` (physical arm). The loop is [scripts/run_real.py](../scripts/run_real.py).

## Phase 1 — validate the plumbing offline (no hardware)

The `SimRobot` backend is a classic-MuJoCo digital twin: it renders the overhead
camera at the training resolution and steps position-controlled physics, so it
exercises the entire deploy path.

```bash
conda activate vision_rl
MUJOCO_GL=osmesa python scripts/run_real.py --robot sim \
    --ckpt <ckpt.msgpack> --episodes 3 --save-video /tmp/loopback.mp4
```

Correctness check (bridge decode + proprio FK vs the real env), should print PASS:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=osmesa python scripts/verify_deploy.py
```

And confirm the policy is actually vision-conditioned before hardware:

```bash
python scripts/check_vision.py --backend cpu --ckpt <ckpt.msgpack>
```

> If `check_vision.py` reports a BLIND policy, sim-to-real is premature — the
> policy ignores the camera and will only ever reproduce one fixed pose. Fix that
> in sim first.

## Phase 2 — physical bring-up (LeRobot)

The SO-101 is a first-class LeRobot robot. `LeRobotSO101` wraps a
`SO101Follower`; adapt `LeRobotSO101.connect(...)` imports to your installed
lerobot version, or pass a follower you already use.

```bash
python scripts/run_real.py --robot real \
    --port /dev/ttyACM0 \        # servos, via LeRobot
    --camera /dev/video0 \       # frames, via OpenCV (-> 84x84 RGB)
    --ckpt <ckpt.msgpack> --unit deg \
    --max-step-rad 0.15          # per-step slew limit — keep low on first runs
```

Joints come from LeRobot (`--port`); camera frames come from the OpenCV
`--camera` device (index `0` or a `/dev/videoN` path), resized to the trained
84×84 exactly like `scripts/camera_84x84.py`.

**Calibration is the make-or-break step** (do this before trusting any motion):

- **Joint zeros/signs/units.** Command each joint to a known angle in sim and on
  the real arm; confirm `read_joints()` (radians) agrees with the sim joint value,
  same sign. LeRobot usually reports **degrees** post-calibration (`--unit deg`);
  if it returns normalized `-100..100`, convert to radians first. A wrong offset
  or sign drives the arm the wrong way.
- **Home pose.** The runner slews to `bridge.home_targets()` before/after a run.
  Verify that pose is safe and matches the sim home visually.
- **Camera pose.** The policy only knows the `overhead_cam` viewpoint
  (`pos="0.16 -0.42 0.34"`, looking at the workspace from −y, ~58° down, in
  [so101_pick_place.xml](../vision_rl/envs/assets/so101_menagerie/so101_pick_place.xml)).
  Mount the real camera to reproduce that view as closely as possible, and match
  the **field of view**. The model input is a **square** 84×84; the bridge resizes
  whatever frame you give it, so feed a square frame (center-crop a 4:3 camera to
  1:1 before it reaches the bridge) or you'll squash the aspect ratio the policy
  learned on.

## Phase 3 — close the vision gap (the hard part)

The policy has only ever seen synthetic renders (solid blue cube, brown table,
checker floor, one light, one camera). A raw webcam frame differs enough that the
CNN will likely not transfer as-is. Recommended, in order:

1. **Match the real scene to the sim** — camera pose/FOV, a blue cube of the right
   size, a plain table, even lighting. Cheapest; try the current policy first.
2. **Domain-randomization retrain** — randomize textures, lighting, camera pose,
   colors, and cube appearance in the renderer, retrain, then deploy. This is what
   makes vision transfer robust. It touches
   [rendering/](../vision_rl/rendering/) and the scene assets, not the deploy code.
3. (Optional) A quick **sim-vs-real image diff**: freeze the real arm, feed its
   joints into `SimRobot`, and compare the rendered frame to the camera frame to
   quantify the gap before retraining.

## Known sim-to-real gaps to expect

- **Velocity** — training used the sim's exact `qvel`; deploy finite-differences
  positions. Keep `ctrl_dt` accurate; low-pass filter if the servo readout is noisy.
- **Actuator dynamics/latency** — the sim `kp=998` position servo is stiffer and
  lag-free vs the real STS3215 (see the note in
  [so101.xml](../vision_rl/envs/assets/so101_menagerie/so101.xml)). Expect softer,
  slower real tracking; the slew limit helps.
- **Contacts/gripper** — real friction and the parallel jaw differ from the sim
  collision model; grasping is the least robust stage.
- **Loop rate** — hit 50 Hz on real hardware (GPU-jitted policy is sub-ms; the
  camera/servo bus is the limiter). The osmesa loopback runs slower purely because
  of CPU rendering, which does not exist on the real robot.
