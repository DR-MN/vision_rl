"""Sim<->real conversion layer for the SO-101 pick-and-place / pick policies.

This is the piece that connects a trained policy to physical hardware. It owns
the *exact* transforms the training env applied, so a policy deployed through it
sees the same observation layout and its actions are decoded the same way it was
trained -- no silent unit/scale mismatch between sim and real. Supports both
`so101_pick_place` and `so101_pick` (cfg.task selects the scene + task config;
both use the same robot, action layout, and proprio layout, so one bridge
implementation covers both).

Two directions:

    model action (6-d, [-1,1])  --action_to_targets-->  joint position targets (rad)
    real joints + camera frame  --observe/reset------->  obs dict {pixels, proprio}

All the magic numbers (home pose, per-joint control ranges, gripper range, joint
order) are read from the same MuJoCo model the env uses, so this file has no
hand-copied constants to drift out of sync. See:
  * action decode      -> envs/so101_pick_place.py: SO101PickPlaceEnv.step (lines ~205-218,
                           incl. the max_joint_delta slew-rate clamp on the arm)
                           (so101_pick.py's step is the same transform, pick-only reward)
  * proprio layout     -> envs/so101_pick_place.py: SO101PickPlaceEnv._proprio
  * frame stacking     -> envs/vision_wrapper.py: _stack_reset / _stack_push

NOTE: any new *stateful, sequential* transform added to either env's step()
(like the slew-rate clamp) must be mirrored here with matching bridge-side
state, or scripts/verify_deploy.py will (correctly) start failing -- see the
2026-07-23 mismatch this file's max_joint_delta clamp was added to fix.
"""

from __future__ import annotations

import numpy as np
import mujoco

from vision_rl.config import Config
from vision_rl.envs.so101_pick_place import _SCENE_XML as _SCENE_XML_PICK_PLACE
from vision_rl.envs.so101_pick_place import _N_ARM, _N_ROBOT
from vision_rl.envs.so101_pick import _SCENE_XML as _SCENE_XML_PICK

_SCENE_XML_BY_TASK = {
    "so101_pick_place": _SCENE_XML_PICK_PLACE,
    "so101_pick": _SCENE_XML_PICK,
}

# Real-servo joint order (matches the MJCF <actuator> order and LeRobot's
# SO101Follower motor order). Exposed so the hardware backend can map 1:1.
JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)


class SO101Bridge:
    """Stateful translator between the policy and the real robot.

    Stateful for two reasons: the observation carries temporal information the
    robot does not (a rolling frame stack and a previous joint position, for
    finite-diff velocity), AND `action_to_targets(...)` now also carries the
    previously-commanded arm target so it can reproduce the env's per-step
    slew-rate limit (SO101PickPlaceEnv/SO101PickEnv.step's `max_joint_delta`
    clamp) -- it is NOT pure. Call `reset(...)` once at episode start (also
    re-seeds the slew reference to home), then `observe(...)` every control step.
    """

    def __init__(self, cfg: Config):
        if cfg.task not in _SCENE_XML_BY_TASK:
            raise ValueError(
                f"SO101Bridge only supports {list(_SCENE_XML_BY_TASK)}, got {cfg.task!r}"
            )
        self.cfg = cfg
        self.task = cfg.task
        # so101_pick_place and so101_pick each have their own config sub-object
        # (cfg.so101 / cfg.pick), but both expose the same field names for
        # everything the bridge needs -- see config.py SO101Config/SO101PickConfig.
        task_cfg = cfg.so101 if cfg.task == "so101_pick_place" else cfg.pick
        self.ctrl_dt = float(task_cfg.ctrl_dt)
        self.action_scale = float(task_cfg.action_scale)
        self.max_joint_delta = float(task_cfg.max_joint_delta)

        # One classic MuJoCo model/data, reused for forward kinematics (grasp
        # site) and for reading the same constants the env baked into training.
        self.model = mujoco.MjModel.from_xml_path(_SCENE_XML_BY_TASK[cfg.task])
        self._fk_data = mujoco.MjData(self.model)

        m = self.model
        self._grasp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        # Home control (arm) + per-joint control ranges + gripper range: identical
        # to what SO101PickPlaceEnv.__init__ pulls out, so the action decode below
        # reproduces training exactly.
        self._home_qpos = np.asarray(m.key_qpos[0], dtype=np.float64)
        home_ctrl = np.asarray(m.key_ctrl[0], dtype=np.float64)
        self._arm_home_ctrl = home_ctrl[:_N_ARM]
        self._home_ctrl = home_ctrl

        # Action decode is centered on `action_center`, a SEPARATE keyframe
        # from `home` (see the env's __init__ and the XML keyframe comment).
        # `home` above is still what home_targets() returns -- the real,
        # hardware-verified pose the arm physically slews to. Read by NAME so
        # this can never silently drift from the training env's own lookup.
        center_key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "action_center")
        self._arm_action_center = np.asarray(m.key_ctrl[center_key][:_N_ARM],
                                              dtype=np.float64)

        ctrlrange = np.asarray(m.actuator_ctrlrange, dtype=np.float64)
        self._arm_low = ctrlrange[:_N_ARM, 0]
        self._arm_high = ctrlrange[:_N_ARM, 1]
        self._grip_low = float(ctrlrange[_N_ARM, 0])
        self._grip_high = float(ctrlrange[_N_ARM, 1])

        # Observation image geometry (must match the checkpoint's render config).
        self.img_h = int(cfg.render.height)
        self.img_w = int(cfg.render.width)
        self.frame_stack = int(cfg.encoder.frame_stack)
        self.camera = cfg.render.camera

        # Rolling state, set up by reset().
        self._frames: np.ndarray | None = None       # [H, W, frame_stack*3] uint8
        self._prev_qpos: np.ndarray | None = None     # [6] rad
        # Slew-rate reference: the env's `state.data.ctrl[:_N_ARM]` equivalent.
        # Initialized to home here (not just in reset()) so action_to_targets
        # works correctly even if called before an explicit reset(), matching
        # the env where ctrl == home_ctrl immediately after construction/reset.
        self._prev_arm_ctrl = self._arm_home_ctrl.copy()

    # ------------------------------------------------------------------ #
    # Spec (for building the policy / hardware backend)
    # ------------------------------------------------------------------ #
    @property
    def action_dim(self) -> int:
        return _N_ARM + 1                              # 5 arm + gripper

    @property
    def proprio_dim(self) -> int:
        return _N_ROBOT + _N_ROBOT + 3                 # qpos(6)+qvel(6)+grasp_xyz(3)

    @property
    def joint_names(self):
        return JOINT_NAMES

    @property
    def gripper_ctrl_range(self) -> tuple[float, float]:
        """(closed, open) gripper joint limits in rad -- for mapping to LeRobot's
        normalized 0..100 gripper command."""
        return (self._grip_low, self._grip_high)

    @property
    def joint_limits_rad(self) -> dict[str, tuple[float, float]]:
        """{joint_name: (low, high)} in rad, in JOINT_NAMES order -- the sim's
        actuator ctrlrange, i.e. what training assumed the real servo range to
        be. Compare against the real robot's calibrated range before trusting
        hardware (see scripts/check_calibration.py): a mismatch means the
        policy can command targets the real arm can't safely reach, or
        `home_targets()` can land somewhere unexpected (e.g. into the table)."""
        limits = {n: (float(lo), float(hi)) for n, lo, hi in
                  zip(JOINT_NAMES[:-1], self._arm_low, self._arm_high)}
        limits[JOINT_NAMES[-1]] = (self._grip_low, self._grip_high)
        return limits

    # ------------------------------------------------------------------ #
    # Model action -> real joint targets
    # ------------------------------------------------------------------ #
    def action_to_targets(self, action: np.ndarray) -> np.ndarray:
        """Decode a policy action into absolute joint position targets (rad).

        Byte-for-byte the transform in SO101PickPlaceEnv.step:
          * arm:     residual around action_center (NOT home -- see the
                     `action_center` keyframe), scaled by action_scale,
                     clipped to range, THEN slew-limited relative to the
                     previously commanded arm target (max_joint_delta rad/step)
          * gripper: [-1,1] -> [closed, open] absolute command (not slew-limited,
                     same as the env)
        The result is what the sim fed to its position actuators, i.e. exactly
        what the real Feetech servos should be commanded to (in radians).

        Stateful: updates `self._prev_arm_ctrl` so the next call's slew clamp
        is relative to THIS step's output, same as the env carrying it in
        `data.ctrl` across steps. Call `reset(...)` at episode start to
        re-seed this to home (the physical start pose -- NOT action_center).
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        arm = self._arm_action_center + self.action_scale * action[:_N_ARM]
        arm = np.clip(arm, self._arm_low, self._arm_high)
        arm = np.clip(arm, self._prev_arm_ctrl - self.max_joint_delta,
                      self._prev_arm_ctrl + self.max_joint_delta)
        self._prev_arm_ctrl = arm

        grip01 = 0.5 * (action[_N_ARM] + 1.0)                  # 0..1
        grip = self._grip_low + grip01 * (self._grip_high - self._grip_low)

        return np.concatenate([arm, [grip]]).astype(np.float64)

    def home_targets(self) -> np.ndarray:
        """The home control vector (arm home + gripper open), in radians."""
        return self._home_ctrl.copy()

    # ------------------------------------------------------------------ #
    # Real joints + camera -> model observation
    # ------------------------------------------------------------------ #
    def grasp_xyz(self, qpos6: np.ndarray) -> np.ndarray:
        """World position of the `gripperframe` site via forward kinematics.

        Uses the SAME model + site the env used, so the 3 proprio grasp-position
        numbers match training. Only the 6 robot joints affect this site, so the
        rest of qpos (the cube) is left at home and ignored.
        """
        self._fk_data.qpos[:] = self._home_qpos
        self._fk_data.qpos[:_N_ROBOT] = qpos6
        mujoco.mj_kinematics(self.model, self._fk_data)
        return np.asarray(self._fk_data.site_xpos[self._grasp_site], dtype=np.float32).copy()

    def _proprio(self, qpos6: np.ndarray, qvel6: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.asarray(qpos6, dtype=np.float32),
            np.asarray(qvel6, dtype=np.float32),
            self.grasp_xyz(qpos6),
        ]).astype(np.float32)

    def _preprocess_frame(self, rgb: np.ndarray) -> np.ndarray:
        """Coerce a camera frame to the trained pixel format: uint8 RGB HxW.

        The caller must pass RGB (convert from OpenCV BGR first). Resizes to the
        training resolution if needed; the sim backend already renders at the
        right size, so this is a no-op there.
        """
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.shape[:2] != (self.img_h, self.img_w):
            rgb = _resize_rgb(rgb, self.img_h, self.img_w)
        return rgb

    def reset(self, qpos6: np.ndarray, rgb: np.ndarray) -> dict:
        """Start a new episode: seed the frame stack, velocity reference, and
        the action_to_targets slew reference (all reset to home, like the env)."""
        frame = self._preprocess_frame(rgb)
        # Repeat the first frame frame_stack times along channels (see
        # vision_wrapper._stack_reset).
        self._frames = np.tile(frame, (1, 1, self.frame_stack)).astype(np.uint8)
        self._prev_qpos = np.asarray(qpos6, dtype=np.float64).copy()
        self._prev_arm_ctrl = self._arm_home_ctrl.copy()
        qvel6 = np.zeros(_N_ROBOT, dtype=np.float32)
        return {"pixels": self._frames, "proprio": self._proprio(qpos6, qvel6)}

    def observe(self, qpos6: np.ndarray, rgb: np.ndarray) -> dict:
        """Build the obs for one control step from live joints + camera frame."""
        if self._frames is None or self._prev_qpos is None:
            raise RuntimeError("call reset(...) before observe(...)")

        qpos6 = np.asarray(qpos6, dtype=np.float64)
        # Velocity by finite difference (hardware-agnostic; servo velocity readout
        # is often noisy/unavailable). NOTE: training used the sim's true qvel, so
        # this is one genuine sim2real gap -- keep ctrl_dt accurate and, if noisy,
        # consider low-pass filtering here.
        qvel6 = ((qpos6 - self._prev_qpos) / self.ctrl_dt).astype(np.float32)
        self._prev_qpos = qpos6.copy()

        frame = self._preprocess_frame(rgb)
        # Drop oldest 3 channels, append newest (see vision_wrapper._stack_push).
        self._frames = np.concatenate([self._frames[..., 3:], frame], axis=-1).astype(np.uint8)
        return {"pixels": self._frames, "proprio": self._proprio(qpos6, qvel6)}


def _resize_rgb(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize an HxWx3 uint8 image to (h, w). Prefers cv2, falls back to PIL."""
    try:
        import cv2
        return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        pass
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
    except Exception as e:  # pragma: no cover - only hit on a bare environment
        raise RuntimeError(
            "need opencv-python or Pillow to resize real camera frames to the "
            f"training resolution ({h}x{w})"
        ) from e
