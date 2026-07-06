"""CNN vision encoder.

A Nature-DQN style convolutional stack that maps a stacked RGB image
(H, W, frame_stack * 3) to a fixed-size embedding. Kept deliberately small
so activations fit comfortably in 4 GB of VRAM at 64x64 resolution.
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


def _orthogonal(scale: float = jnp.sqrt(2.0)):
    return nn.initializers.orthogonal(scale)


class VisionEncoder(nn.Module):
    """Convolutional encoder producing a `features`-dim embedding.

    Input is expected as float image(s) in [0, 1] (or raw uint8 if
    `normalize` is set) with a trailing channel axis. Supports an arbitrary
    number of leading batch dims via `jnp.reshape` folding.
    """

    channels: Sequence[int] = (32, 64, 64)
    kernel_sizes: Sequence[int] = (8, 4, 3)
    strides: Sequence[int] = (4, 2, 1)
    features: int = 256
    normalize: bool = True

    @nn.compact
    def __call__(self, pixels: jax.Array) -> jax.Array:
        # Fold leading batch dims (e.g. [T, B, H, W, C]) into one.
        *batch_dims, h, w, c = pixels.shape
        x = pixels.reshape((-1, h, w, c))

        if self.normalize:
            # Accept uint8 or float; scale to [0, 1] if it looks like 0..255.
            x = x.astype(jnp.float32)
            x = jnp.where(x.max() > 1.5, x / 255.0, x)

        for ch, k, s in zip(self.channels, self.kernel_sizes, self.strides):
            x = nn.Conv(
                features=ch,
                kernel_size=(k, k),
                strides=(s, s),
                padding="VALID",
                kernel_init=_orthogonal(),
            )(x)
            x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))               # flatten conv maps
        x = nn.Dense(self.features, kernel_init=_orthogonal())(x)
        x = nn.relu(x)

        return x.reshape((*batch_dims, self.features))
