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
    """Drive a physical SO-101 through a `lerobot` follower object.

    lerobot's module paths and value units shift between releases, so rather than
    hard-wire a version this backend wraps a follower you construct and connect
    yourself (or use `LeRobotSO101.connect(...)` best-effort). We only translate:

        obs["{joint}.pos"] (native units)  <-- joint angles -->  action dict
        obs[camera_key]                    <-- RGB frame

    CALIBRATION (must verify before trusting the arm):
      * `unit`: lerobot commonly reports degrees after calibration; set "rad" if
        yours already returns radians. If it returns NORMALIZED values (-100..100),
        convert those to radians before passing in, or extend this class.
      * The sim's joint zeros/signs must line up with the servo calibration. Move
        each joint to a known angle and confirm read_joints() matches the sim.
        A wrong sign or offset here will drive the arm the wrong way -- start with
        a slew-rate limit and be ready on the e-stop (see scripts/run_real.py).
    """

    def __init__(self, robot, joint_names, camera_key: str,
                 unit: str = "deg", bgr: bool = False):
        self.robot = robot
        self.joint_names = tuple(joint_names)
        self.camera_key = camera_key
        self.bgr = bgr
        if unit not in ("deg", "rad"):
            raise ValueError("unit must be 'deg' or 'rad'")
        self._to_rad = (np.pi / 180.0) if unit == "deg" else 1.0
        self._from_rad = (180.0 / np.pi) if unit == "deg" else 1.0
        if not getattr(robot, "is_connected", False):
            robot.connect()

    @classmethod
    def connect(cls, port: str, robot_id: str = "so101", camera_key: str = "overhead",
                camera_index: int = 0, fps: int = 30, width: int = 640, height: int = 480,
                unit: str = "deg", bgr: bool = False):
        """Best-effort construction of an SO101Follower. Adapt imports to your
        installed lerobot version if these paths differ."""
        try:                                            # newer layout
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
            from lerobot.cameras.opencv import OpenCVCameraConfig
        except Exception:                               # older layout
            from lerobot.common.robots.so101_follower import (  # type: ignore
                SO101Follower, SO101FollowerConfig)
            from lerobot.common.cameras.opencv import OpenCVCameraConfig  # type: ignore

        cam_cfg = {camera_key: OpenCVCameraConfig(
            index_or_path=camera_index, fps=fps, width=width, height=height)}
        robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, cameras=cam_cfg))
        from vision_rl.deploy.bridge import JOINT_NAMES
        return cls(robot, JOINT_NAMES, camera_key, unit=unit, bgr=bgr)

    def _read_obs(self) -> dict:
        return self.robot.get_observation()

    def reset(self) -> np.ndarray:
        return self.read_joints()

    def read_joints(self) -> np.ndarray:
        obs = self._read_obs()
        vals = [float(obs[f"{name}.pos"]) for name in self.joint_names]
        return np.asarray(vals, dtype=np.float64) * self._to_rad

    def read_camera(self) -> np.ndarray:
        obs = self._read_obs()
        frame = np.asarray(obs[self.camera_key])
        if self.bgr:
            frame = frame[..., ::-1]                    # BGR -> RGB
        return frame

    def send_targets(self, targets_rad: np.ndarray) -> None:
        targets = np.asarray(targets_rad, dtype=np.float64) * self._from_rad
        action = {f"{name}.pos": float(v)
                  for name, v in zip(self.joint_names, targets)}
        self.robot.send_action(action)

    def home(self) -> None:
        pass  # the runner slews to home_targets() before disconnecting

    def close(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:
            pass
