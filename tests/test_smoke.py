"""End-to-end smoke tests: shapes and one training iteration on the CPU renderer.

These are intentionally tiny so they run on any machine (no Madrona needed).
Run with:  python -m tests.test_smoke
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vision_rl.config import small_config
from vision_rl.envs import VisionVecEnv
from vision_rl.models import ActorCritic
from vision_rl.train import make_train


def _tiny_cfg():
    cfg = small_config()
    cfg.ppo.num_envs = 4
    cfg.ppo.rollout_length = 4
    cfg.ppo.num_minibatches = 2
    cfg.render.backend = "cpu"
    cfg.render.width = 48
    cfg.render.height = 48
    return cfg


def test_env_reset_step_shapes():
    cfg = _tiny_cfg()
    env = VisionVecEnv(cfg)
    rng = jax.random.PRNGKey(0)
    vstate = env.reset(rng)

    b = cfg.ppo.num_envs
    assert vstate.obs["pixels"].shape == (b, *env.pixels_shape)
    assert vstate.obs["proprio"].shape == (b, env.proprio_size)
    assert vstate.obs["pixels"].dtype == jnp.uint8

    actions = jnp.zeros((b, env.action_size))
    vstate2 = env.step(vstate, actions)
    assert vstate2.obs["pixels"].shape == (b, *env.pixels_shape)
    assert vstate2.env_state.reward.shape == (b,)
    print("[ok] env reset/step shapes")


def test_network_forward():
    cfg = _tiny_cfg()
    env = VisionVecEnv(cfg)
    net = ActorCritic(action_dim=env.action_size)
    params = net.init(jax.random.PRNGKey(0), env.sample_obs())

    obs = {
        "pixels": jnp.zeros((8, *env.pixels_shape), jnp.uint8),
        "proprio": jnp.zeros((8, env.proprio_size), jnp.float32),
    }
    pi, value = net.apply(params, obs)
    assert pi.sample(seed=jax.random.PRNGKey(1)).shape == (8, env.action_size)
    assert value.shape == (8,)
    print("[ok] network forward")


def test_one_training_iteration():
    cfg = _tiny_cfg()
    cfg.ppo.total_env_steps = cfg.batch_size  # exactly one update
    train = make_train(cfg)
    out = train()
    assert len(out["history"]) >= 1
    print("[ok] one training iteration:", out["history"][-1])


if __name__ == "__main__":
    test_env_reset_step_shapes()
    test_network_forward()
    test_one_training_iteration()
    print("\nAll smoke tests passed.")
