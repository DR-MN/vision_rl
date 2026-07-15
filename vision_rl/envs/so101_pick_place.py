"""Single-robot SO-101 pick-and-place environment on MuJoCo-JAX (MJX).

The SO-101 (5-DoF arm + 1-DoF parallel-ish jaw gripper) must pick a cube and
deliver it to a target pad. As with the Franka reach task, the cube and target
positions are *not* in the proprioceptive observation -- the agent must locate
them from the camera image (added by the vision wrapper).

Action (6-d, all in [-1, 1]):
    [0:5] residual position targets for the 5 arm joints
    [5]   gripper command (mapped to the jaw's ctrl range: -1=closed, +1=open)

Reward is staged and dense: reach the cube -> lift it -> carry it over the
target -> bonus for a successful placement. Written for a single world; the
vision wrapper adds vmap + rendering + frame-stacking + auto-reset.
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import mjx

from vision_rl.config import SO101Config

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "so101_menagerie")
_SCENE_XML = os.path.join(_ASSET_DIR, "so101_pick_place.xml")

_N_ARM = 5           # controlled arm joints
_N_ROBOT = 6         # arm + gripper joints (qpos 0..5); cube addr found dynamically


@struct.dataclass
class SO101State:
    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    target_pos: jax.Array    # [3] place-target world position
    cube_init_z: jax.Array   # scalar resting height (for lift measurement)
    was_lifted: jax.Array    # scalar 0/1, latched once cube is lifted
    grasped: jax.Array       # scalar 0/1, cube currently attached to gripper
    prev_reach: jax.Array    # last-step gripper->cube distance (progress shaping)
    prev_place: jax.Array    # last-step cube->pad distance (progress shaping)
    step_count: jax.Array
    rng: jax.Array
    metrics: dict[str, Any]


class SO101PickPlaceEnv:
    def __init__(self, cfg: SO101Config | None = None, impl: str = "jax",
                 naconmax: int = 64):
        self.cfg = cfg or SO101Config()
        self.impl = impl                              # "jax" (default) or "warp"
        self._naconmax = naconmax                     # warp global contact buffer

        self.mj_model = mujoco.MjModel.from_xml_path(_SCENE_XML)
        self.mjx_model = (mjx.put_model(self.mj_model, impl="warp")
                          if impl == "warp" else mjx.put_model(self.mj_model))
        self.xml_path = _SCENE_XML

        sim_dt = float(self.mj_model.opt.timestep)
        self.n_substeps = max(1, round(self.cfg.ctrl_dt / sim_dt))

        m = self.mj_model
        self._grasp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        target_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "place_target")
        self._target_mocap = int(m.body_mocapid[target_body])

        # Cube free-joint qpos/qvel addresses (robust to body ordering).
        cube_jnt = m.body_jntadr[cube_body]
        self._cube_qadr = int(m.jnt_qposadr[cube_jnt])          # 7 numbers (pos+quat)
        self._cube_vadr = int(m.jnt_dofadr[cube_jnt])           # 6 numbers (lin+ang)

        self._home_qpos = jnp.asarray(m.key_qpos[0])
        self._home_ctrl = jnp.asarray(m.key_ctrl[0])
        self._arm_home_ctrl = self._home_ctrl[:_N_ARM]

        ctrlrange = jnp.asarray(m.actuator_ctrlrange)
        self._arm_ctrl_low = ctrlrange[:_N_ARM, 0]
        self._arm_ctrl_high = ctrlrange[:_N_ARM, 1]
        self._grip_low = float(ctrlrange[_N_ARM, 0])
        self._grip_high = float(ctrlrange[_N_ARM, 1])

        self._cube_low = jnp.asarray(self.cfg.cube_low)
        self._cube_high = jnp.asarray(self.cfg.cube_high)
        self._tgt_low = jnp.asarray(self.cfg.target_low)
        self._tgt_high = jnp.asarray(self.cfg.target_high)

        # Table-contact penalty: the MJCF now has collision geoms on the arm, so
        # the solver physically stops it at the table on BOTH backends. The warp
        # backend does not expose data.contact to Python, so the *reward* signal
        # uses the grasp-site height instead (see _table_dive). Clearance is
        # measured above the table top at z=0.
        self._table_clear = 0.03

    # ------------------------------------------------------------------ #
    @property
    def action_size(self) -> int:
        return _N_ARM + 1                      # arm + gripper

    @property
    def proprio_size(self) -> int:
        # robot qpos (6) + robot qvel (6) + grasp-site xyz (3)
        return _N_ROBOT + _N_ROBOT + 3

    # ------------------------------------------------------------------ #
    def _make_data(self):
        # Warp impl requires the MjModel (not the mjx Model) + explicit impl.
        # naconmax sizes the per-world contact buffer (table+cube need headroom
        # or you get "narrowphase overflow" and dropped contacts).
        if self.impl == "warp":
            return mjx.make_data(self.mj_model, impl="warp", naconmax=self._naconmax)
        return mjx.make_data(self.mjx_model)

    def _cube_pos(self, data):
        return data.qpos[self._cube_qadr : self._cube_qadr + 3]

    def _grasp_pos(self, data):
        return data.site_xpos[self._grasp_site]

    def _proprio(self, data):
        return jnp.concatenate([
            data.qpos[:_N_ROBOT],
            data.qvel[:_N_ROBOT],
            self._grasp_pos(data),
        ])

    def _table_dive(self, data, d_reach):
        """Backend-agnostic proxy for the gripper diving into the table.

        The warp Data does not expose `data.contact`, so instead of reading the
        contact buffer we use the grasp-site height: a dive is when the gripper
        is below the table clearance while NOT near the cube (i.e. pressing into
        the table somewhere other than the object it is meant to pick). Uses only
        the grasp-site position and reach distance, both available on jax + warp.
        """
        grasp_z = self._grasp_pos(data)[2]
        below = jnp.clip(self._table_clear - grasp_z, 0.0, None)
        away = (d_reach >= self.cfg.grasp_dist).astype(jnp.float32)
        return ((below > 0.005) & (away > 0.5)).astype(jnp.float32)

    # ------------------------------------------------------------------ #
    def reset(self, rng: jax.Array) -> SO101State:
        rng, q_rng, c_rng, t_rng = jax.random.split(rng, 4)

        qpos = self._home_qpos.at[:_N_ARM].add(
            0.05 * jax.random.normal(q_rng, (_N_ARM,))
        )
        # Randomize cube xy in the spawn region.
        cube_xy = jax.random.uniform(
            c_rng, (2,), minval=self._cube_low, maxval=self._cube_high
        )
        qpos = qpos.at[self._cube_qadr : self._cube_qadr + 2].set(cube_xy)
        qpos = qpos.at[self._cube_qadr + 2].set(self.cfg.cube_z)

        data = self._make_data()
        data = data.replace(qpos=qpos, ctrl=self._home_ctrl)

        # Randomize the place target.
        target_xy = jax.random.uniform(
            t_rng, (2,), minval=self._tgt_low, maxval=self._tgt_high
        )
        target_pos = jnp.array([target_xy[0], target_xy[1], 0.001])
        mocap_pos = data.mocap_pos.at[self._target_mocap].set(target_pos)
        data = data.replace(mocap_pos=mocap_pos)

        data = mjx.forward(self.mjx_model, data)

        obs = self._proprio(data)
        cube0 = self._cube_pos(data)
        d_reach = jnp.linalg.norm(self._grasp_pos(data) - cube0)
        d_place = jnp.linalg.norm(cube0[:2] - target_pos[:2])
        metrics = {"dist": d_reach, "success": jnp.float32(0.0),
                   "cube_z": cube0[2], "lifted": jnp.float32(0.0),
                   "table_hit": jnp.float32(0.0)}
        return SO101State(
            data=data, obs=obs,
            reward=jnp.float32(0.0), done=jnp.float32(0.0),
            target_pos=target_pos, cube_init_z=jnp.float32(self.cfg.cube_z),
            was_lifted=jnp.float32(0.0), grasped=jnp.float32(0.0),
            prev_reach=d_reach, prev_place=d_place,
            step_count=jnp.int32(0),
            rng=rng, metrics=metrics,
        )

    def step(self, state: SO101State, action: jax.Array) -> SO101State:
        action = jnp.clip(action, -1.0, 1.0)

        # Arm: residual position control around home.
        arm_ctrl = self._arm_home_ctrl + self.cfg.action_scale * action[:_N_ARM]
        arm_ctrl = jnp.clip(arm_ctrl, self._arm_ctrl_low, self._arm_ctrl_high)
        # Gripper: map [-1,1] -> [closed, open] absolute command.
        grip = 0.5 * (action[_N_ARM] + 1.0)                     # 0..1
        grip_ctrl = self._grip_low + grip * (self._grip_high - self._grip_low)
        ctrl = state.data.ctrl.at[:_N_ARM].set(arm_ctrl)
        ctrl = ctrl.at[_N_ARM].set(grip_ctrl)
        data = state.data.replace(ctrl=ctrl)

        def _substep(d, _):
            return mjx.step(self.mjx_model, d), None
        data, _ = jax.lax.scan(_substep, data, None, length=self.n_substeps)

        grasp = self._grasp_pos(data)
        cube = self._cube_pos(data)                      # real physics, no assist
        d_reach = jnp.linalg.norm(grasp - cube)

        cube_lift = cube[2] - state.cube_init_z
        # Whether the cube is CURRENTLY off the table (physically held/lifted).
        lifted_now = (cube[2] > self.cfg.lift_thresh).astype(jnp.float32)
        # Latch: a real pick happened at some point this episode (success needs it).
        ever_lifted = jnp.maximum(state.was_lifted, lifted_now)
        d_place = jnp.linalg.norm(cube[:2] - state.target_pos[:2])

        placed = (ever_lifted > 0) & (d_place < self.cfg.success_thresh) & \
                 (cube[2] < state.cube_init_z + 0.02)
        success = placed.astype(jnp.float32)

        # --- Re-grasp on slip ---
        # Reach is active whenever the cube is neither currently held nor already
        # placed, so a dropped/slipped cube pulls the arm back to pick it up
        # again. Place shaping is active only while the cube is truly lifted.
        reach_gate = 1.0 - jnp.maximum(lifted_now, success)
        place_gate = lifted_now

        reach_progress = state.prev_reach - d_reach
        place_progress = place_gate * (state.prev_place - d_place)

        # Gripper-closing command (action[5] < 0 -> closing); shaping so the jaw
        # tends to close when the fingers are actually at the cube.
        grip_closing = (0.5 * (action[_N_ARM] + 1.0) < 0.5).astype(jnp.float32)
        near_cube = (d_reach < self.cfg.grasp_dist).astype(jnp.float32)

        reach_r = -self.cfg.reach_scale * d_reach * reach_gate
        reach_pr = self.cfg.reach_progress_scale * reach_progress * reach_gate
        grasp_r = self.cfg.grasp_bonus_scale * near_cube * grip_closing * reach_gate
        # The physical lift signal: the cube only rises if it is really gripped.
        lift_r = self.cfg.lift_scale * jnp.clip(cube_lift, 0.0, self.cfg.max_lift)
        place_r = place_gate * self.cfg.place_scale * (
            1.0 - jnp.clip(d_place, 0.0, self.cfg.place_radius) / self.cfg.place_radius
        )
        place_pr = self.cfg.place_progress_scale * place_progress
        succ_r = self.cfg.success_bonus * success
        ctrl_cost = self.cfg.ctrl_cost_scale * jnp.sum(jnp.square(action))
        # Penalize the gripper diving into the table away from the cube.
        table_hit = self._table_dive(data, d_reach)
        table_pen = self.cfg.table_penalty_scale * table_hit
        reward = (reach_r + reach_pr + grasp_r + lift_r + place_r + place_pr
                  + succ_r - ctrl_cost - table_pen)

        step_count = state.step_count + 1
        done = (step_count >= self.cfg.episode_length).astype(jnp.float32)

        obs = self._proprio(data)
        metrics = {"dist": d_reach, "success": success, "cube_z": cube[2],
                   "lifted": lifted_now, "table_hit": table_hit}
        return state.replace(
            data=data, obs=obs, reward=reward, done=done,
            was_lifted=ever_lifted, grasped=lifted_now,
            prev_reach=d_reach, prev_place=d_place,
            step_count=step_count, metrics=metrics,
        )
