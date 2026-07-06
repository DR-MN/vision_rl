"""Minimal diagonal Gaussian policy distribution.

Hand-rolled to avoid a heavy/fragile distrax + tensorflow-probability
dependency. Covers exactly what PPO needs: sample, log_prob, entropy.
All quantities are summed over the action dimension (independent dims).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_LOG_2PI = jnp.log(2.0 * jnp.pi)


class DiagGaussian:
    """Diagonal (independent) multivariate normal over the last axis."""

    def __init__(self, mean: jax.Array, std: jax.Array):
        self.mean = mean
        self.std = std

    def sample(self, seed: jax.Array) -> jax.Array:
        eps = jax.random.normal(seed, self.mean.shape)
        return self.mean + self.std * eps

    def log_prob(self, value: jax.Array) -> jax.Array:
        var = jnp.square(self.std)
        log_det = 2.0 * jnp.log(self.std)
        per_dim = -0.5 * (jnp.square(value - self.mean) / var + log_det + _LOG_2PI)
        return jnp.sum(per_dim, axis=-1)

    def entropy(self) -> jax.Array:
        per_dim = 0.5 * (_LOG_2PI + 1.0) + jnp.log(self.std)
        return jnp.sum(per_dim, axis=-1)

    def mode(self) -> jax.Array:
        return self.mean
