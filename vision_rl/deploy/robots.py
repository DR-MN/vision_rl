"""Hardware backends behind one interface, so the control loop is identical
whether it drives a physical SO-101 or a MuJoCo loopback.

    RobotBackend            abstract contract
    SimRobot                classic-MuJoCo digital twin (no hardware needed)
    LeRobotSO101            real SO-101 via the `lerobot` SO101Follower API

The interface mirrors what a real position-controlled arm gives you:
    reset()            -> settle to home, return current joint angles (rad)
    read_joints()      -> the 6 joint angles (rad), servo order = bridge.JOINT_NAMES
    read_camera()      -> an RGB uint8 frame (HxWx3)
    send_targets(rad)  -> command 6 absolute joint position targets (rad)
    home() / close()   -> safe shutdown

Units are radians everywhere at this boundary; each backend converts to/from its
own native units internally.
"""

from __future__ import annotations

import abc

import numpy as np
import mujoco

from vision_rl.config import Config
from vision_rl.envs.so101_pick_place import _SCENE_XML, _N_ARM, _N_ROBOT


class RobotBackend(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> np.ndarray: ...
    @abc.abstractmethod
    def read_joints(self) -> np.ndarray: ...
    @abc.abstractmethod
    def read_camera(self) -> np.ndarray: ...
    @abc.abstractmethod
    def send_targets(self, targets_rad: np.ndarray) -> None: ...

    def home(self) -> None:  # optional
        pass

    def close(self) -> None:  # optional
        pass


# --------------------------------------------------------------------------- #
# Digital twin: classic MuJoCo, no hardware. Validates the full bridge/policy
# plumbing and lets you watch the trained policy attempt the task offline.
# --------------------------------------------------------------------------- #
class SimRobot(RobotBackend):
    """A MuJoCo stand-in for the real arm.

    Deliberately uses the *classic* engine (not MJX) and re-derives the scene
    from the XML, so it is an independent check on the deploy path rather than a
    re-run of the training env. Physics is stepped `n_substeps` per control step,
    matching ctrl_dt, and the overhead camera is rendered at the training
    resolution -- the best case for the vision policy (pixels ~identical to what
    it trained on).
    """

    def __init__(self, cfg: Config, seed: int = 0, randomize: bool = True):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(_SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self._rng = np.random.default_rng(seed)
        self._randomize = randomize

        sim_dt = float(self.model.opt.timestep)
        self.n_substeps = max(1, round(float(cfg.so101.ctrl_dt) / sim_dt))

        m = self.model
        self._home_qpos = np.asarray(m.key_qpos[0], dtype=np.float64)
        self._home_ctrl = np.asarray(m.key_ctrl[0], dtype=np.float64)

        cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._cube_qadr = int(m.jnt_qposadr[m.body_jntadr[cube_body]])
        tgt_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "place_target")
        self._tgt_mocap = int(m.body_mocapid[tgt_body])

        self._cube_low = np.asarray(cfg.so101.cube_low)
        self._cube_high = np.asarray(cfg.so101.cube_high)
        self._tgt_low = np.asarray(cfg.so101.target_low)
        self._tgt_high = np.asarray(cfg.so101.target_high)
        self._cube_z = float(cfg.so101.cube_z)

        self._cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cfg.render.camera)
        if self._cam < 0:
            raise ValueError(f"camera '{cfg.render.camera}' not in model")
        self._renderer = mujoco.Renderer(
            m, height=int(cfg.render.height), width=int(cfg.render.width)
        )

    def reset(self) -> np.ndarray:
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] = self._home_qpos
        if self._randomize:
            d.qpos[:_N_ARM] += 0.05 * self._rng.standard_normal(_N_ARM)
            cube_xy = self._rng.uniform(self._cube_low, self._cube_high)
            d.qpos[self._cube_qadr:self._cube_qadr + 2] = cube_xy
            d.qpos[self._cube_qadr + 2] = self._cube_z
            tgt_xy = self._rng.uniform(self._tgt_low, self._tgt_high)
            d.mocap_pos[self._tgt_mocap] = [tgt_xy[0], tgt_xy[1], 0.001]
        d.ctrl[:] = self._home_ctrl
        mujoco.mj_forward(self.model, d)
        return self.read_joints()

    def read_joints(self) -> np.ndarray:
        return np.asarray(self.data.qpos[:_N_ROBOT], dtype=np.float64).copy()

    def read_camera(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render()               # uint8 RGB [H, W, 3]

    def send_targets(self, targets_rad: np.ndarray) -> None:
        self.data.ctrl[:] = np.asarray(targets_rad, dtype=np.float64)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

    def close(self) -> None:
        try:
            self._renderer.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Real hardware via LeRobot's SO101Follower.
# --------------------------------------------------------------------------- #
class LeRobotSO101(RobotBackend):
    """Drive a physical SO-101: joints via a `lerobot` follower, frames via an
    external `camera` object (e.g. deploy.camera.OpenCVCamera).

    The camera is decoupled from LeRobot on purpose: the policy needs 84x84 RGB
    from a fixed external viewpoint, which the OpenCV camera produces directly
    (same as scripts/camera_84x84.py). LeRobot is used only for the servos.

    Units (verified against lerobot SO101Follower):
      * With `use_degrees=True` (set by `connect`), the 5 ARM joints read/accept
        **degrees** -> converted to/from radians here.
      * The GRIPPER is always `RANGE_0_100` (0=closed .. 100=open in the servo's
        calibrated range), NOT degrees. Mapped to/from the sim gripper joint range
        (`grip_range`, in rad) linearly. Use `grip_flip=True` if your calibration
        has 0=open.

    CALIBRATION (must verify before trusting the arm):
      * Run LeRobot's servo calibration first, or read_joints() is meaningless.
      * The sim's joint zeros/signs must line up with the servo calibration. Move
        each joint to a known angle and confirm read_joints() (rad) matches the
        sim pose. A wrong sign/offset drives the arm the wrong way -- `offset_rad`
        lets you correct per-joint zero offsets without editing code. Start with a
        slew-rate limit and be ready on the e-stop (see scripts/run_real.py).
    """

    def __init__(self, robot, joint_names, camera, grip_range,
                 arm_unit: str = "deg", offset_rad=None, grip_flip: bool = False):
        self.robot = robot
        self.joint_names = tuple(joint_names)           # 6 names, gripper last
        self.camera = camera                            # object with .read()/.close()
        self._grip_low, self._grip_high = float(grip_range[0]), float(grip_range[1])
        self._grip_flip = grip_flip
        if arm_unit not in ("deg", "rad"):
            raise ValueError("arm_unit must be 'deg' or 'rad'")
        self._arm_to_rad = (np.pi / 180.0) if arm_unit == "deg" else 1.0
        self._arm_from_rad = (180.0 / np.pi) if arm_unit == "deg" else 1.0
        self._offset = (np.zeros(len(self.joint_names)) if offset_rad is None
                        else np.asarray(offset_rad, dtype=np.float64))
        if not getattr(robot, "is_connected", False):
            robot.connect()

    @property
    def _arm_names(self):
        return self.joint_names[:-1]

    @property
    def _grip_name(self):
        return self.joint_names[-1]

    def _grip_norm_to_rad(self, norm: float) -> float:
        f = (100.0 - norm) if self._grip_flip else norm
        return self._grip_low + (f / 100.0) * (self._grip_high - self._grip_low)

    def _grip_rad_to_norm(self, rad: float) -> float:
        frac = (rad - self._grip_low) / (self._grip_high - self._grip_low)
        norm = float(np.clip(frac * 100.0, 0.0, 100.0))
        return (100.0 - norm) if self._grip_flip else norm

    @classmethod
    def connect(cls, port: str, camera, grip_range, robot_id: str = "so101",
                offset_rad=None, grip_flip: bool = False):
        """Construct an SO101Follower (servos only; frames come from `camera`).
        Forces `use_degrees=True` so the arm reports degrees."""
        try:                                            # current layout (src/lerobot)
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except Exception:                               # older layout
            from lerobot.common.robots.so101_follower import (  # type: ignore
                SO101Follower, SO101FollowerConfig)

        robot = SO101Follower(SO101FollowerConfig(
            port=port, id=robot_id, cameras={}, use_degrees=True))
        from vision_rl.deploy.bridge import JOINT_NAMES
        return cls(robot, JOINT_NAMES, camera, grip_range,
                   arm_unit="deg", offset_rad=offset_rad, grip_flip=grip_flip)

    def reset(self) -> np.ndarray:
        return self.read_joints()

    def read_joints(self) -> np.ndarray:
        obs = self.robot.get_observation()
        arm = np.asarray([float(obs[f"{n}.pos"]) for n in self._arm_names],
                         dtype=np.float64) * self._arm_to_rad
        grip = self._grip_norm_to_rad(float(obs[f"{self._grip_name}.pos"]))
        return np.concatenate([arm, [grip]]) + self._offset

    def read_camera(self) -> np.ndarray:
        return self.camera.read()                       # uint8 RGB [H, W, 3]

    def send_targets(self, targets_rad: np.ndarray) -> None:
        t = np.asarray(targets_rad, dtype=np.float64) - self._offset
        arm_deg = t[:len(self._arm_names)] * self._arm_from_rad
        action = {f"{n}.pos": float(v) for n, v in zip(self._arm_names, arm_deg)}
        action[f"{self._grip_name}.pos"] = self._grip_rad_to_norm(float(t[-1]))
        self.robot.send_action(action)

    def close(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:
            pass
        if self.camera is not None:
            self.camera.close()
