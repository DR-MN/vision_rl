"""PPO training loop for vision-based Franka reach on MJX.

Puts the pieces together:
    VisionVecEnv (MJX physics + batch render + frame stack)
        -> ActorCritic (CNN encoder + Gaussian policy + value)
        -> rollout collection (jitted lax.scan)
        -> GAE -> PPO minibatch updates.

Run with `python -m vision_rl.train` or via `scripts/train_franka_reach.py`.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from vision_rl.config import Config
from vision_rl.envs import VisionVecEnv
from vision_rl.models import ActorCritic
from vision_rl.ppo import compute_gae, create_train_state, ppo_update


def _build_network(cfg: Config, action_size: int) -> ActorCritic:
    return ActorCritic(
        action_dim=action_size,
        encoder_channels=cfg.encoder.channels,
        encoder_kernels=cfg.encoder.kernel_sizes,
        encoder_strides=cfg.encoder.strides,
        encoder_features=cfg.encoder.features,
        normalize_pixels=cfg.encoder.normalize_pixels,
        init_log_std=cfg.ppo.init_log_std,
    )


def make_train(cfg: Config) -> Callable[[], dict]:
    env = VisionVecEnv(cfg)
    network = _build_network(cfg, env.action_size)
    ppo = cfg.ppo

    num_updates = ppo.total_env_steps // cfg.batch_size

    def apply_fn(params, obs):
        return network.apply(params, obs)

    def params_init(rng):
        return network.init(rng, env.sample_obs())

    # ----------------------------------------------------------------- #
    # Rollout collection: T sequential steps across B envs.
    # ----------------------------------------------------------------- #
    def collect_rollout(train_state, vstate, rng):
        def _step(carry, _):
            vstate, rng = carry
            rng, akey = jax.random.split(rng)
            pi, value = apply_fn(train_state.params, vstate.obs)
            action = pi.sample(seed=akey)
            log_prob = pi.log_prob(action)

            next_vstate = env.step(vstate, action)
            transition = {
                "obs": vstate.obs,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": next_vstate.env_state.reward,
                "done": next_vstate.env_state.done,
                "dist": next_vstate.env_state.metrics["dist"],
                "success": next_vstate.env_state.metrics["success"],
            }
            return (next_vstate, rng), transition

        (vstate, rng), traj = jax.lax.scan(
            _step, (vstate, rng), None, length=ppo.rollout_length
        )
        # Bootstrap value for the state after the last collected step.
        _, last_value = apply_fn(train_state.params, vstate.obs)
        return vstate, traj, last_value, rng

    # ----------------------------------------------------------------- #
    # One full iteration: collect -> GAE -> flatten -> PPO update.
    # ----------------------------------------------------------------- #
    @jax.jit
    def train_iteration(train_state, vstate, rng):
        rng, collect_rng, update_rng = jax.random.split(rng, 3)
        vstate, traj, last_value, _ = collect_rollout(
            train_state, vstate, collect_rng
        )

        advantages, returns = compute_gae(
            traj["reward"], traj["value"], traj["done"],
            last_value, ppo.gamma, ppo.gae_lambda,
        )

        # Flatten [T, B, ...] -> [T*B, ...].
        def _flat(x):
            return x.reshape((-1,) + x.shape[2:])

        batch = {
            "obs": jax.tree_util.tree_map(_flat, traj["obs"]),
            "actions": _flat(traj["action"]),
            "log_probs": _flat(traj["log_prob"]),
            "values": _flat(traj["value"]),
            "advantages": _flat(advantages),
            "returns": _flat(returns),
        }
        train_state, metrics = ppo_update(train_state, batch, update_rng, ppo)

        metrics = {
            **metrics,
            "mean_reward": traj["reward"].mean(),
            "mean_dist": traj["dist"].mean(),
            "success_rate": traj["success"].mean(),
        }
        return train_state, vstate, metrics

    # ----------------------------------------------------------------- #
    # Driver
    # ----------------------------------------------------------------- #
    def train():
        rng = jax.random.PRNGKey(ppo.seed)
        rng, init_rng, reset_rng = jax.random.split(rng, 3)

        train_state = create_train_state(
            init_rng, apply_fn, params_init, ppo, num_updates
        )
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(train_state.params))
        print(f"[init] renderer backend = {env.renderer.backend}")
        print(f"[init] policy params = {n_params:,}")
        print(f"[init] updates = {num_updates}, batch = {cfg.batch_size}, "
              f"minibatch = {cfg.minibatch_size}")

        vstate = env.reset(reset_rng)

        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        history = []
        start = time.time()
        for it in range(1, num_updates + 1):
            rng, it_rng = jax.random.split(rng)
            train_state, vstate, metrics = train_iteration(
                train_state, vstate, it_rng
            )

            if it % ppo.log_interval == 0:
                metrics = jax.tree_util.tree_map(lambda x: float(x), metrics)
                history.append({"iter": it, **metrics})
                steps = it * cfg.batch_size
                sps = steps / (time.time() - start)
                print(
                    f"[{it:>5}/{num_updates}] steps={steps:>10,} "
                    f"R={metrics['mean_reward']:+.3f} "
                    f"dist={metrics['mean_dist']:.3f} "
                    f"succ={metrics['success_rate']:.2f} "
                    f"kl={metrics['approx_kl']:.4f} "
                    f"ent={metrics['entropy']:+.2f} "
                    f"| {sps:,.0f} sps"
                )

            if it % ppo.ckpt_interval == 0 or it == num_updates:
                path = os.path.join(cfg.ckpt_dir, f"{cfg.exp_name}_it{it}.msgpack")
                with open(path, "wb") as f:
                    f.write(serialization.to_bytes(train_state.params))

        return {"train_state": train_state, "history": history}

    return train


def main():
    import argparse

    from vision_rl.config import small_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="tiny config for tests")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--backend", type=str, default=None,
                        choices=["madrona", "cpu", "auto"])
    args = parser.parse_args()

    cfg = small_config() if args.small else Config()
    if args.num_envs is not None:
        cfg.ppo.num_envs = args.num_envs
    if args.steps is not None:
        cfg.ppo.total_env_steps = args.steps
    if args.backend is not None:
        cfg.render.backend = args.backend

    train = make_train(cfg)
    train()


if __name__ == "__main__":
    main()
