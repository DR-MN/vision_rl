"""Actor-critic network on top of the vision encoder.

Observation is a dict:
    - "pixels":  (H, W, frame_stack * 3) image
    - "proprio": (proprio_dim,) vector of joint pos/vel etc.

The policy is a diagonal Gaussian with a state-independent, learnable
log-std (standard for continuous-control PPO). The value head is a scalar.
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from vision_rl.models.distributions import DiagGaussian
from vision_rl.models.encoder import VisionEncoder


def _orthogonal(scale: float):
    return nn.initializers.orthogonal(scale)


class ActorCritic(nn.Module):
    action_dim: int
    encoder_channels: Sequence[int] = (32, 64, 64)
    encoder_kernels: Sequence[int] = (8, 4, 3)
    encoder_strides: Sequence[int] = (4, 2, 1)
    encoder_features: int = 256
    normalize_pixels: bool = True
    hidden_sizes: Sequence[int] = (256, 256)
    init_log_std: float = -0.5
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs: dict) -> tuple[DiagGaussian, jax.Array]:
        encoder = VisionEncoder(
            channels=self.encoder_channels,
            kernel_sizes=self.encoder_kernels,
            strides=self.encoder_strides,
            features=self.encoder_features,
            normalize=self.normalize_pixels,
            layer_norm=self.layer_norm,
        )
        img_embed = encoder(obs["pixels"])

        proprio = obs.get("proprio")
        if proprio is not None:
            h = jnp.concatenate([img_embed, proprio], axis=-1)
        else:
            h = img_embed

        # Shared trunk, then separate actor / critic heads. tanh saturates for
        # |pre-activation| >~ 4, where its derivative underflows and no gradient
        # reaches the encoder, so normalise before squashing.
        for size in self.hidden_sizes:
            h = nn.Dense(size, kernel_init=_orthogonal(jnp.sqrt(2.0)))(h)
            if self.layer_norm:
                h = nn.LayerNorm()(h)
            h = nn.tanh(h)

        # Actor: small init on the last layer keeps early actions near zero.
        mean = nn.Dense(self.action_dim, kernel_init=_orthogonal(0.01))(h)
        log_std = self.param(
            "log_std",
            lambda _key, shape: jnp.full(shape, self.init_log_std),
            (self.action_dim,),
        )
        std = jnp.exp(log_std) * jnp.ones_like(mean)
        pi = DiagGaussian(mean=mean, std=std)

        # Critic.
        value = nn.Dense(1, kernel_init=_orthogonal(1.0))(h)
        value = jnp.squeeze(value, axis=-1)

        return pi, value
