"""Single-robot Franka reach environment on MuJoCo-JAX (MJX).

The robot must move its gripper to a randomly-placed target. The target's
position is *not* part of the proprioceptive observation -- the agent has to
locate it from the camera image (added by the vision wrapper). This makes the
task genuinely vision-dependent.

The env is written for a single world; parallelism and rendering are added by
`VisionVecEnv` (vmap + batch renderer). Physics-only logic lives here so it can
be tested and reasoned about in isolation.

State/obs contract of this class:
    reset(rng) -> EnvState
    step(state, action) -> EnvState
where obs is the proprio vector only; pixels are attached downstream.
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import mjx

from vision_rl.config import EnvConfig

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "franka_emika_panda")
_SCENE_XML = os.path.join(_ASSET_DIR, "franka_reach.xml")

_N_ARM = 7  # controlled arm joints (gripper held fixed)


@struct.dataclass
class EnvState:
    """Per-env simulation + bookkeeping state (batched via vmap)."""

    data: mjx.Data           # MJX physics state
    obs: jax.Array           # proprio observation
    reward: jax.Array        # scalar
    done: jax.Array          # scalar (1.0 terminal/truncated)
    target_pos: jax.Array    # [3] world-frame target position
    step_count: jax.Array    # int, steps since reset
    rng: jax.Array           # PRNG key carried for in-env randomness
    metrics: dict[str, Any]  # logging (dist, success, ...)


class FrankaReachEnv:
    """MJX Franka reach. Instances are cheap; the heavy model is shared."""

    def __init__(self, cfg: EnvConfig | None = None, impl: str = "jax",
                 naconmax: int = 64, njmax: int = 128):
        self.cfg = cfg or EnvConfig()

        self.impl = impl                              # "jax" (default) or "warp"
        self._naconmax = naconmax
        self._njmax = njmax                           # warp per-world constraint buffer
        self.mj_model = mujoco.MjModel.from_xml_path(_SCENE_XML)
        self.mjx_model = (mjx.put_model(self.mj_model, impl="warp")
                          if impl == "warp" else mjx.put_model(self.mj_model))
        self.xml_path = _SCENE_XML

        # Control decimation: policy steps at ctrl_dt, physics at model timestep.
        sim_dt = float(self.mj_model.opt.timestep)
        self.n_substeps = max(1, round(self.cfg.ctrl_dt / sim_dt))

        # --- cache indices / constants -----------------------------------
        m = self.mj_model
        self._gripper_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        target_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap = int(m.body_mocapid[target_body])

        # Home keyframe (index 0) gives neutral qpos/ctrl.
        self._home_qpos = jnp.asarray(m.key_qpos[0])
        self._home_ctrl = jnp.asarray(m.key_ctrl[0])
        self._arm_home_ctrl = self._home_ctrl[:_N_ARM]

        # Actuator control ranges for the arm (for clipping residual targets).
        ctrlrange = jnp.asarray(m.actuator_ctrlrange)  # [nu, 2]
        self._arm_ctrl_low = ctrlrange[:_N_ARM, 0]
        self._arm_ctrl_high = ctrlrange[:_N_ARM, 1]

        self._target_low = jnp.asarray(self.cfg.target_low)
        self._target_high = jnp.asarray(self.cfg.target_high)

    # ------------------------------------------------------------------ #
    # Spaces
    # ------------------------------------------------------------------ #
    @property
    def action_size(self) -> int:
        return _N_ARM

    @property
    def proprio_size(self) -> int:
        # arm qpos (7) + arm qvel (7) + gripper xyz (3)
        return _N_ARM + _N_ARM + 3

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _proprio(self, data: mjx.Data) -> jax.Array:
        arm_qpos = data.qpos[:_N_ARM]
        arm_qvel = data.qvel[:_N_ARM]
        gripper_xpos = data.site_xpos[self._gripper_site]
        return jnp.concatenate([arm_qpos, arm_qvel, gripper_xpos])

    def _gripper_pos(self, data: mjx.Data) -> jax.Array:
        return data.site_xpos[self._gripper_site]

    # ------------------------------------------------------------------ #
    # Reset / step
    # ------------------------------------------------------------------ #
    def _make_data(self):
        if self.impl == "warp":
            return mjx.make_data(self.mj_model, impl="warp",
                                  naconmax=self._naconmax, njmax=self._njmax)
        return mjx.make_data(self.mjx_model)

    def reset(self, rng: jax.Array) -> EnvState:
        rng, q_rng, t_rng = jax.random.split(rng, 3)

        # Small noise on the home pose so episodes differ.
        qpos = self._home_qpos.at[:_N_ARM].add(
            0.05 * jax.random.normal(q_rng, (_N_ARM,))
        )
        data = self._make_data()
        data = data.replace(qpos=qpos, ctrl=self._home_ctrl)

        # Randomize the visible target inside the workspace volume.
        target_pos = jax.random.uniform(
            t_rng, (3,), minval=self._target_low, maxval=self._target_high
        )
        mocap_pos = data.mocap_pos.at[self._target_mocap].set(target_pos)
        data = data.replace(mocap_pos=mocap_pos)

        data = mjx.forward(self.mjx_model, data)  # populate site_xpos etc.

        obs = self._proprio(data)
        metrics = {"dist": jnp.linalg.norm(self._gripper_pos(data) - target_pos),
                   "success": jnp.float32(0.0),
                   "reward": jnp.float32(0.0)}
        return EnvState(
            data=data,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            target_pos=target_pos,
            step_count=jnp.int32(0),
            rng=rng,
            metrics=metrics,
        )

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        # Residual position control: target = home + scaled action, clipped.
        action = jnp.clip(action, -1.0, 1.0)
        arm_ctrl = self._arm_home_ctrl + self.cfg.action_scale * action
        arm_ctrl = jnp.clip(arm_ctrl, self._arm_ctrl_low, self._arm_ctrl_high)
        ctrl = state.data.ctrl.at[:_N_ARM].set(arm_ctrl)
        data = state.data.replace(ctrl=ctrl)

        # Advance physics n_substeps times.
        def _substep(d, _):
            return mjx.step(self.mjx_model, d), None

        data, _ = jax.lax.scan(_substep, data, None, length=self.n_substeps)

        gripper_pos = self._gripper_pos(data)
        dist = jnp.linalg.norm(gripper_pos - state.target_pos)
        success = (dist < self.cfg.success_threshold).astype(jnp.float32)

        reward = (
            -self.cfg.reach_reward_scale * dist
            + self.cfg.success_bonus * success
            - self.cfg.ctrl_cost_scale * jnp.sum(jnp.square(action))
        )

        step_count = state.step_count + 1
        done = (step_count >= self.cfg.episode_length).astype(jnp.float32)

        obs = self._proprio(data)
        metrics = {"dist": dist, "success": success, "reward": reward}
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            step_count=step_count,
            metrics=metrics,
        )
