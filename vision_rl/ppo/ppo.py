"""Custom PPO: train-state construction, clipped loss, minibatch update.

Everything here is pure/jittable. The training loop (train.py) owns the
environment interaction and calls `ppo_update` once per iteration on the
flattened rollout batch.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from vision_rl.config import PPOConfig


class PPOTrainState(TrainState):
    """Standard Flax TrainState (params + optimizer). Aliased for clarity."""


def create_train_state(
    rng: jax.Array,
    apply_fn: Callable,
    params_init_fn: Callable[[jax.Array], Any],
    cfg: PPOConfig,
    num_updates: int,
) -> PPOTrainState:
    """Initialise params and the optimizer (grad-clip + optional LR anneal)."""
    params = params_init_fn(rng)

    if cfg.anneal_lr:
        # One "decay unit" per gradient step across the whole run.
        total_grad_steps = num_updates * cfg.update_epochs * cfg.num_minibatches
        lr = optax.linear_schedule(cfg.learning_rate, 0.0, total_grad_steps)
    else:
        lr = cfg.learning_rate

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )
    return PPOTrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def _ppo_loss(
    params,
    apply_fn,
    obs,
    actions,
    old_log_probs,
    advantages,
    returns,
    old_values,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
):
    pi, values = apply_fn(params, obs)
    log_probs = pi.log_prob(actions)
    entropy = pi.entropy().mean()

    # Clipped policy-gradient objective.
    ratio = jnp.exp(log_probs - old_log_probs)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Clipped value loss (PPO2-style).
    v_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    vf_loss1 = jnp.square(values - returns)
    vf_loss2 = jnp.square(v_clipped - returns)
    vf_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2).mean()

    loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy

    approx_kl = ((ratio - 1.0) - jnp.log(ratio)).mean()
    clip_frac = (jnp.abs(ratio - 1.0) > clip_eps).mean()
    metrics = {
        "loss": loss,
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }
    return loss, metrics


def ppo_update(
    state: PPOTrainState,
    batch: dict,
    rng: jax.Array,
    cfg: PPOConfig,
) -> tuple[PPOTrainState, dict]:
    """Run `update_epochs` of minibatch SGD over a flattened rollout batch.

    `batch` keys (all leading dim = num_envs * rollout_length = N):
        obs (pytree), actions, log_probs, advantages, returns, values.

    Not jitted on its own (cfg is a plain dataclass); it is called inside the
    jitted `train_iteration`, where cfg's fields fold in as trace-time
    constants.
    """
    n = batch["advantages"].shape[0]
    mb_size = n // cfg.num_minibatches

    # Replace any non-finite advantages/returns (from a blown-up env) with 0 so
    # they contribute nothing rather than propagating NaNs into the gradient.
    batch = {**batch,
             "advantages": jnp.nan_to_num(batch["advantages"]),
             "returns": jnp.nan_to_num(batch["returns"])}
    if cfg.normalize_advantages:
        adv = batch["advantages"]
        batch = {**batch, "advantages": (adv - adv.mean()) / (adv.std() + 1e-8)}

    grad_fn = jax.value_and_grad(_ppo_loss, has_aux=True)

    def _minibatch_step(carry, mb_idx):
        state = carry
        idx = mb_idx  # [mb_size] indices into the shuffled batch
        mb = jax.tree_util.tree_map(lambda x: x[idx], batch)
        (_, metrics), grads = grad_fn(
            state.params,
            state.apply_fn,
            mb["obs"],
            mb["actions"],
            mb["log_probs"],
            mb["advantages"],
            mb["returns"],
            mb["values"],
            cfg.clip_eps,
            cfg.entropy_coef,
            cfg.value_coef,
        )
        # NaN guard: if a physics blow-up makes any gradient non-finite, zero it
        # so that bad batch is skipped instead of poisoning the whole network
        # (those envs recover on their next auto-reset).
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    def _epoch(carry, epoch_rng):
        state = carry
        perm = jax.random.permutation(epoch_rng, n)
        # Drop the remainder so every minibatch is full, then reshape.
        perm = perm[: mb_size * cfg.num_minibatches]
        mb_indices = perm.reshape((cfg.num_minibatches, mb_size))
        state, metrics = jax.lax.scan(_minibatch_step, state, mb_indices)
        return state, metrics

    epoch_rngs = jax.random.split(rng, cfg.update_epochs)
    state, metrics = jax.lax.scan(_epoch, state, epoch_rngs)
    # Average metrics over all epochs & minibatches.
    metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
    return state, metrics
