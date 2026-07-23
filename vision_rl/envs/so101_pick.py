"""Single-robot SO-101 pick-only environment on MuJoCo-JAX (MJX).

Same robot/scene as the pick-and-place task (envs/so101_pick_place.py), but
the goal is just to pick the cube up and hold it above `cfg.pick_height`
(10 cm above the table by default) -- no carry-to-target/place stage. As with
the pick-and-place task, the cube position is *not* in the proprioceptive
observation -- the agent must locate it from the camera image (added by the
vision wrapper).

Action (6-d, all in [-1, 1]):
    [0:5] residual position targets for the 5 arm joints
    [5]   gripper command (mapped to the jaw's ctrl range: -1=closed, +1=open)

Reward is staged and dense: reach the cube -> grasp it -> lift it above
pick_height. Written for a single world; the vision wrapper adds vmap +
rendering + frame-stacking + auto-reset.
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import mjx

from vision_rl.config import SO101PickConfig
from vision_rl.envs.buffers import size_buffers
from vision_rl.envs.so101_pick_place import _N_ARM, _N_ROBOT

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "so101_menagerie")
_SCENE_XML = os.path.join(_ASSET_DIR, "so101_pick.xml")


@struct.dataclass
class SO101PickState:
    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    cube_init_z: jax.Array   # scalar resting height (for lift measurement)
    was_picked: jax.Array    # scalar 0/1, latched once a successful pick happens
    grasped: jax.Array       # scalar 0/1, cube currently off the table
    prev_reach: jax.Array    # last-step gripper->cube distance (progress shaping)
    step_count: jax.Array
    rng: jax.Array
    metrics: dict[str, Any]


class SO101PickEnv:
    def __init__(self, cfg: SO101PickConfig | None = None, impl: str = "jax",
                 num_envs: int = 1, naconmax: int | None = None,
                 njmax: int | None = None):
        self.cfg = cfg or SO101PickConfig()
        self.impl = impl                              # "jax" (default) or "warp"

        self.mj_model = mujoco.MjModel.from_xml_path(_SCENE_XML)
        # Warp allocates contact/constraint storage up front and silently drops
        # the overflow, so size it from this scene (None => probe + headroom).
        # Only the warp impl reads these; skip the probe cost on the jax path.
        self._naconmax, self._njmax = (
            size_buffers(self.mj_model, num_envs, naconmax, njmax)
            if impl == "warp" else (naconmax or 0, njmax or 0))
        self.mjx_model = (mjx.put_model(self.mj_model, impl="warp")
                          if impl == "warp" else mjx.put_model(self.mj_model))
        self.xml_path = _SCENE_XML

        sim_dt = float(self.mj_model.opt.timestep)
        self.n_substeps = max(1, round(self.cfg.ctrl_dt / sim_dt))

        m = self.mj_model
        self._grasp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")

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

        # Table-contact penalty: see so101_pick_place.SO101PickPlaceEnv for the
        # backend-agnostic rationale (warp doesn't expose data.contact).
        # Clearance is measured above the table top at z=0.
        self._table_clear = self.cfg.table_clear

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
        # Buffers are sized in __init__; see envs/buffers.py for the scoping
        # (naconmax is global, njmax is per-world) and why undersizing is silent.
        if self.impl == "warp":
            return mjx.make_data(self.mj_model, impl="warp",
                                  naconmax=self._naconmax, njmax=self._njmax)
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
        """Backend-agnostic proxy for the gripper touching/diving into the table.

        See so101_pick_place.SO101PickPlaceEnv._table_dive for the full
        rationale; entering the clearance zone above the table top while away
        from the cube counts as a hit.
        """
        grasp_z = self._grasp_pos(data)[2]
        below = jnp.clip(self._table_clear - grasp_z, 0.0, None)
        away = (d_reach >= self.cfg.grasp_dist).astype(jnp.float32)
        return ((below > 0.0) & (away > 0.5)).astype(jnp.float32)

    # ------------------------------------------------------------------ #
    def reset(self, rng: jax.Array) -> SO101PickState:
        rng, q_rng, c_rng = jax.random.split(rng, 3)

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
        data = mjx.forward(self.mjx_model, data)

        obs = self._proprio(data)
        cube0 = self._cube_pos(data)
        d_reach = jnp.linalg.norm(self._grasp_pos(data) - cube0)
        metrics = {"dist": d_reach, "success": jnp.float32(0.0),
                   "cube_z": cube0[2], "lifted": jnp.float32(0.0),
                   "table_hit": jnp.float32(0.0)}
        return SO101PickState(
            data=data, obs=obs,
            reward=jnp.float32(0.0), done=jnp.float32(0.0),
            cube_init_z=jnp.float32(self.cfg.cube_z),
            was_picked=jnp.float32(0.0), grasped=jnp.float32(0.0),
            prev_reach=d_reach,
            step_count=jnp.int32(0),
            rng=rng, metrics=metrics,
        )

    def step(self, state: SO101PickState, action: jax.Array) -> SO101PickState:
        action = jnp.clip(action, -1.0, 1.0)

        # Arm: residual position control around home.
        arm_ctrl = self._arm_home_ctrl + self.cfg.action_scale * action[:_N_ARM]
        arm_ctrl = jnp.clip(arm_ctrl, self._arm_ctrl_low, self._arm_ctrl_high)
        # Slew-rate limit: cap the per-step change in commanded joint target so
        # the arm can't snap to a new pose in one 20ms control step (mirrors the
        # real-deploy slew limit in scripts/run_real.py). Slows the arm down
        # without shrinking its reachable range.
        prev_arm_ctrl = state.data.ctrl[:_N_ARM]
        arm_ctrl = jnp.clip(arm_ctrl, prev_arm_ctrl - self.cfg.max_joint_delta,
                             prev_arm_ctrl + self.cfg.max_joint_delta)
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

        # Gripper-closing command (action[5] < 0 -> closing); shaping so the jaw
        # tends to close when the fingers are actually at the cube.
        grip_closing = (0.5 * (action[_N_ARM] + 1.0) < 0.5).astype(jnp.float32)
        near_cube = (d_reach < self.cfg.grasp_dist).astype(jnp.float32)
        # "Held" proxy: fingers at the cube AND jaw commanded closed. Keeps the
        # grasp bonus alive through the lift so reach->grasp->lift has no
        # reward cliff.
        held = near_cube * grip_closing

        # Pick succeeds once the cube is genuinely held AND raised above
        # pick_height (10cm above the table by default).
        picked = held * (cube[2] > self.cfg.pick_height).astype(jnp.float32)
        success = picked
        ever_picked = jnp.maximum(state.was_picked, success)

        # Reach shaping is active whenever the cube is neither currently held
        # nor already successfully picked, so a dropped/slipped cube pulls the
        # arm back to pick it up again.
        reach_gate = 1.0 - jnp.maximum(lifted_now, success)

        reach_progress = state.prev_reach - d_reach

        reach_r = -self.cfg.reach_scale * d_reach * reach_gate
        reach_pr = self.cfg.reach_progress_scale * reach_progress * reach_gate
        # Grasp bonus stays on while the cube is held, even above the lift line.
        grasp_r = self.cfg.grasp_bonus_scale * held
        # The physical lift signal: the cube only rises if it is really gripped.
        # Caps at pick_height -- the reward saturates once the goal height is hit.
        lift_r = self.cfg.lift_scale * jnp.clip(cube_lift, 0.0, self.cfg.pick_height)
        succ_r = self.cfg.success_bonus * success
        ctrl_cost = self.cfg.ctrl_cost_scale * jnp.sum(jnp.square(action))
        # Penalize the gripper touching/diving into the table away from the cube.
        table_hit = self._table_dive(data, d_reach)
        table_pen = self.cfg.table_penalty_scale * table_hit
        reward = (reach_r + reach_pr + grasp_r + lift_r
                  + succ_r - ctrl_cost - table_pen)

        step_count = state.step_count + 1
        done = (step_count >= self.cfg.episode_length).astype(jnp.float32)

        obs = self._proprio(data)
        metrics = {"dist": d_reach, "success": success, "cube_z": cube[2],
                   "lifted": lifted_now, "table_hit": table_hit}
        return state.replace(
            data=data, obs=obs, reward=reward, done=done,
            was_picked=ever_picked, grasped=lifted_now,
            prev_reach=d_reach,
            step_count=step_count, metrics=metrics,
        )
