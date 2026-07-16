"""Tiled live viewer: show many training worlds at once in one MuJoCo window.

A MuJoCo viewer renders a single simulation world, so to watch all parallel
envs we build a *display-only* model that tiles N copies of the scene in a grid
(via `MjSpec.attach`) and drive every tile from the batched training state each
frame.

Only used for visualization — physics/training still run in MJX on the GPU; here
we just `mj_forward` a classic model to pose it, then `viewer.sync()`.
"""

from __future__ import annotations

import math
import time

import mujoco
import mujoco.viewer
import numpy as np


class TiledMirror:
    def __init__(
        self,
        xml_path: str,
        single_model: mujoco.MjModel,
        n_show: int,
        spacing: float = 0.7,
        realtime: bool = True,
        ctrl_dt: float = 0.02,
        launch: bool = True,
    ):
        self.n = n_show
        self.cols = int(math.ceil(math.sqrt(n_show)))
        self._realtime = realtime
        self._dt = ctrl_dt
        self._nq1 = single_model.nq

        # Grid offsets per instance.
        offs = np.zeros((n_show, 3), dtype=np.float64)
        for i in range(n_show):
            r, c = divmod(i, self.cols)
            offs[i] = [c * spacing, r * spacing, 0.0]
        self._offs = offs

        # Build the tiled display model.
        parent = mujoco.MjSpec()
        parent.option.timestep = float(single_model.opt.timestep)
        # One shared floor + light (children's are stripped to avoid z-fighting
        # coplanar planes and over-bright stacked lights).
        _light = parent.worldbody.add_light(pos=[spacing, spacing, 3.0],
                                            dir=[0, 0, -1])
        try:  # prefer a directional light if this build exposes the enum
            _light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        except Exception:
            pass
        print(f"[gui] light type set: {_light.type}") 
        # Match the scene's ground height so tiled tables/legs rest on it.
        fid = mujoco.mj_name2id(single_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        floor_z = float(single_model.geom_pos[fid][2]) if fid >= 0 else 0.0
        floor = parent.worldbody.add_geom()
        floor.name = "tiled_floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.pos = [0.0, 0.0, floor_z]
        floor.size = [0, 0, 0.05]
        floor.rgba = [0.3, 0.32, 0.35, 1.0]

        for i in range(n_show):
            child = mujoco.MjSpec.from_file(xml_path)
            # Hide each child's floor (avoid coplanar z-fighting) and disable its
            # lights (avoid N stacked lights washing out the scene).
            for g in list(child.geoms):
                if g.name == "floor":
                    g.rgba = [0, 0, 0, 0]
            for lt in list(child.lights):
                try:
                    lt.active = 0
                except Exception:
                    pass
            frame = parent.worldbody.add_frame()
            frame.pos = list(offs[i])
            parent.attach(child, prefix=f"w{i}_", frame=frame)

        self.model = parent.compile()
        self.data = mujoco.MjData(self.model)

        # --- precompute additive qpos offset (tile shift for free-joint roots) ---
        qadd = np.zeros(self.model.nq, dtype=np.float64)
        free_adr = [int(single_model.jnt_qposadr[j])
                    for j in range(single_model.njnt)
                    if single_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        for i in range(n_show):
            for adr in free_adr:
                qadd[i * self._nq1 + adr: i * self._nq1 + adr + 3] += offs[i]
        self._qadd = qadd

        # --- mocap mapping: single mocap body names -> tiled mocap ids ---
        self._nmocap1 = single_model.nmocap
        single_mocap_names = []
        for b in range(single_model.nbody):
            if single_model.body_mocapid[b] >= 0:
                single_mocap_names.append(
                    (int(single_model.body_mocapid[b]),
                     mujoco.mj_id2name(single_model, mujoco.mjtObj.mjOBJ_BODY, b)))
        single_mocap_names.sort()  # by single mocap index
        self._mocap_ids = np.full((n_show, self._nmocap1), -1, dtype=np.int64)
        for i in range(n_show):
            for s, name in single_mocap_names:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                        f"w{i}_{name}")
                self._mocap_ids[i, s] = int(self.model.body_mocapid[bid])

        self._viewer = None
        if launch:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print(f"[gui] tiled viewer open — showing {n_show} worlds. "
                  "Close the window to stop mirroring.")

    # ------------------------------------------------------------------ #
    @property
    def alive(self) -> bool:
        return self._viewer is None or self._viewer.is_running()

    def set_frame(self, qpos_b: np.ndarray, mocap_b: np.ndarray):
        """Pose the tiled model from one frame of the batch.

        qpos_b:  [>=n, nq_single]     mocap_b: [>=n, nmocap_single, 3]
        """
        n = self.n
        self.data.qpos[:] = qpos_b[:n].reshape(-1) + self._qadd
        if self._nmocap1:
            ids = self._mocap_ids.reshape(-1)
            vals = (mocap_b[:n] + self._offs[:, None, :]).reshape(-1, 3)
            self.data.mocap_pos[ids] = vals
        mujoco.mj_forward(self.model, self.data)

    def replay(self, viz: dict):
        """Replay a rollout: viz['qpos'] [T,B,nq], viz['mocap'] [T,B,nmocap,3]."""
        if not self.alive:
            return
        qpos = np.asarray(viz["qpos"])
        mocap = np.asarray(viz["mocap"])
        for t in range(qpos.shape[0]):
            if not self.alive:
                return
            self.set_frame(qpos[t], mocap[t])
            if self._viewer is not None:
                self._viewer.sync()
            if self._realtime:
                time.sleep(self._dt)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
