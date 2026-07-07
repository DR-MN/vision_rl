"""Vectorised vision environment wrapper.

Wraps `FrankaReachEnv` to:
  * run `num_envs` worlds in parallel via `jax.vmap`,
  * render a camera image for every world with the batch renderer,
  * stack the last `frame_stack` frames (temporal information for the CNN),
  * auto-reset finished episodes so rollout collection never stalls.

The observation returned to the agent is a dict::
    {"pixels": uint8 [B, H, W, frame_stack*3], "proprio": float [B, proprio_dim]}
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from vision_rl.config import Config
from vision_rl.rendering import make_renderer


@struct.dataclass
class VecState:
    env_state: EnvState          # batched [B, ...]
    frames: jax.Array            # uint8 [B, H, W, frame_stack*3]
    render_token: object         # opaque renderer token (Madrona) or None
    obs: dict                    # {"pixels", "proprio"}
    rng: jax.Array               # key for auto-reset randomness


class VisionVecEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.num_envs = cfg.ppo.num_envs
        self.frame_stack = cfg.encoder.frame_stack

        from vision_rl.envs import make_env  # lazy import to avoid import cycle

        self.env = make_env(cfg)
        self.renderer = make_renderer(
            self.env.mj_model, self.env.mjx_model, cfg.render, self.num_envs
        )

        self._vreset = jax.vmap(self.env.reset)
        self._vstep = jax.vmap(self.env.step)

    # ------------------------------------------------------------------ #
    # Spec
    # ------------------------------------------------------------------ #
    @property
    def mj_model(self):
        """Classic MjModel of the base env (for the live GUI viewer)."""
        return self.env.mj_model

    @property
    def xml_path(self) -> str:
        """Scene XML path of the base env (for the tiled GUI viewer)."""
        return self.env.xml_path

    @property
    def ctrl_dt(self) -> float:
        return self.env.cfg.ctrl_dt

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def pixels_shape(self):
        return (self.cfg.render.height, self.cfg.render.width, self.frame_stack * 3)

    @property
    def proprio_size(self) -> int:
        return self.env.proprio_size

    def sample_obs(self) -> dict:
        """A zero observation with correct shapes (for network init)."""
        return {
            "pixels": jnp.zeros((1, *self.pixels_shape), dtype=jnp.uint8),
            "proprio": jnp.zeros((1, self.proprio_size), dtype=jnp.float32),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _where_done(done, x_reset, x_keep):
        """Select reset vs continued leaf per env (done broadcast over dims)."""
        mask = done.reshape((done.shape[0],) + (1,) * (x_keep.ndim - 1))
        return jnp.where(mask, x_reset, x_keep)

    def _stack_reset(self, rgb):
        # Repeat the first frame `frame_stack` times along channels.
        return jnp.tile(rgb, (1, 1, 1, self.frame_stack))

    def _stack_push(self, frames, rgb):
        # Drop oldest 3 channels, append newest frame.
        return jnp.concatenate([frames[..., 3:], rgb], axis=-1)

    # ------------------------------------------------------------------ #
    # Reset / step
    # ------------------------------------------------------------------ #
    def reset(self, rng: jax.Array) -> VecState:
        rng, sub = jax.random.split(rng)
        keys = jax.random.split(sub, self.num_envs)
        env_state = self._vreset(keys)

        token, rgb = self.renderer.init(env_state.data)      # rgb [B,H,W,3]
        frames = self._stack_reset(rgb)
        obs = {"pixels": frames, "proprio": env_state.obs}
        return VecState(env_state, frames, token, obs, rng)

    def step(self, vstate: VecState, actions: jax.Array) -> VecState:
        stepped = self._vstep(vstate.env_state, actions)
        # Keep the transition's reward/done before auto-reset overwrites state.
        reward, done = stepped.reward, stepped.done

        # Auto-reset finished worlds.
        rng, sub = jax.random.split(vstate.rng)
        reset_keys = jax.random.split(sub, self.num_envs)
        reset_state = self._vreset(reset_keys)
        env_state = jax.tree_util.tree_map(
            lambda r, k: self._where_done(done, r, k), reset_state, stepped
        )
        env_state = env_state.replace(reward=reward, done=done)

        # Render post-reset data so frames match the (possibly new) state.
        token, rgb = self.renderer.render(vstate.render_token, env_state.data)

        frames = jnp.where(
            done.reshape((-1, 1, 1, 1)) > 0,
            self._stack_reset(rgb),
            self._stack_push(vstate.frames, rgb),
        )
        obs = {"pixels": frames, "proprio": env_state.obs}
        return VecState(env_state, frames, token, obs, rng)
