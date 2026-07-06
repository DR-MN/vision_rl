"""Generalised Advantage Estimation (GAE).

Works on rollouts shaped [T, B] (time-major), which is what the training
loop produces from `T` sequential steps across `B` parallel environments.
Bootstrapping uses the value of the state *after* the last collected step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_gae(
    rewards: jax.Array,        # [T, B]
    values: jax.Array,         # [T, B]  value of s_t
    dones: jax.Array,          # [T, B]  1.0 if s_{t+1} is terminal/truncated
    last_value: jax.Array,     # [B]     value bootstrap for s_T
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Return (advantages, returns), both shaped [T, B]."""

    def _step(carry, xs):
        gae, next_value = carry
        reward, value, done = xs
        not_done = 1.0 - done
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, value), gae

    # Scan backwards over time.
    (_, _), advantages = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns
