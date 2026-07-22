"""Real-time policy inference for deployment.

Loads a training checkpoint and exposes a single-observation, batch-1
`act(obs)` that returns the deterministic action (distribution mode). Reuses the
exact network builder (`vision_rl.train._build_network`) and checkpoint loader
(`vision_rl.checkpoint`) so the deployed weights are identical to training --
only the batch dimension and the sampling (mode vs sample) differ.

The forward pass is jitted once and then called ~50x/s in the control loop.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from vision_rl import checkpoint as ckpt_io
from vision_rl.train import _build_network


class RealPolicy:
    def __init__(self, ckpt_path: str, action_dim: int, proprio_dim: int):
        ck = ckpt_io.load(ckpt_path)
        self.cfg = ck.config
        self.step = ck.step

        h, w = int(self.cfg.render.height), int(self.cfg.render.width)
        c = int(self.cfg.encoder.frame_stack) * 3
        self._pixels_shape = (h, w, c)
        self._proprio_dim = proprio_dim

        net = _build_network(self.cfg, action_dim)
        sample = {
            "pixels": jnp.zeros((1, h, w, c), dtype=jnp.uint8),
            "proprio": jnp.zeros((1, proprio_dim), dtype=jnp.float32),
        }
        params = ck.restore_params(net.init(jax.random.PRNGKey(0), sample))
        self._params = params

        @jax.jit
        def _act(params, pixels, proprio):
            obs = {"pixels": pixels[None], "proprio": proprio[None]}
            pi, value = net.apply(params, obs)
            return pi.mode()[0], value[0]

        self._act = _act
        # Warm up the JIT so the first control step isn't a multi-second stall.
        self._act(self._params,
                  jnp.zeros(self._pixels_shape, jnp.uint8),
                  jnp.zeros(proprio_dim, jnp.float32))

    def act(self, obs: dict) -> np.ndarray:
        """Deterministic action for one observation dict {pixels, proprio}."""
        pixels = jnp.asarray(obs["pixels"], dtype=jnp.uint8)
        proprio = jnp.asarray(obs["proprio"], dtype=jnp.float32)
        action, _ = self._act(self._params, pixels, proprio)
        return np.asarray(action)

    def act_with_value(self, obs: dict) -> tuple[np.ndarray, float]:
        """Action plus the critic's value estimate (handy for logging)."""
        pixels = jnp.asarray(obs["pixels"], dtype=jnp.uint8)
        proprio = jnp.asarray(obs["proprio"], dtype=jnp.float32)
        action, value = self._act(self._params, pixels, proprio)
        return np.asarray(action), float(value)
