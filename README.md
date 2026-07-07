# Vision-based RL with PPO on MuJoCo-JAX

A compact, from-scratch pipeline that trains a **single Franka Panda arm** to
**reach a randomly-placed target from camera pixels**, using **PPO** with a
**CNN vision encoder**, on the **MuJoCo-JAX (MJX)** physics engine.

Everything runs in JAX: physics (MJX), rendering (Madrona batch renderer, with a
CPU fallback), the policy/value network (Flax), and PPO (Optax). Rollouts are
fully jitted end-to-end when the GPU renderer is available.

```
                 ┌──────────────────────────────────────────────────────┐
                 │                   VisionVecEnv  (B worlds)             │
   action ──────▶│  MJX physics step ─▶ batch render ─▶ frame stack (k)  │──▶ obs
                 │  (FrankaReachEnv)     (Madrona/CPU)                    │   { pixels[H,W,3k],
                 └──────────────────────────────────────────────────────┘     proprio[d] }
                                             │
                                             ▼
                         ┌───────────────────────────────────┐
                obs ────▶│  ActorCritic                       │
                         │   pixels ─▶ CNN VisionEncoder ─┐    │
                         │                                ├─▶ MLP trunk ─┬─▶ π  (Gaussian, μ, logσ)
                         │   proprio ─────────────────────┘             └─▶ V  (value)
                         └───────────────────────────────────┘
                                             │
                          rollout ─▶ GAE ─▶ PPO clipped update (Optax)
```
## Project layout

```
vision_rl/
├── config.py                 # all hyper-parameters (dataclasses)
├── envs/
│   ├── franka_reach.py       # single-world MJX env: Franka reach
│   ├── so101_pick_place.py   # single-world MJX env: SO-101 pick-and-place
│   ├── vision_wrapper.py     # vmap + batch render + frame stack + auto-reset
│   ├── __init__.py           # make_env(cfg) task factory
│   └── assets/
│       ├── franka_emika_panda/franka_reach.xml
│       └── so101/            # SO-101 model (converted from URDF) + scene
├── rendering/
│   └── batch_renderer.py     # Madrona (GPU) and CPU rendering backends
├── models/
│   ├── encoder.py            # CNN vision encoder
│   ├── distributions.py      # hand-rolled diagonal Gaussian (no distrax)
│   └── networks.py           # actor-critic (Gaussian policy + value)
├── ppo/
│   ├── gae.py                # generalised advantage estimation
│   └── ppo.py                # train state, clipped loss, minibatch update
└── train.py                  # rollout collection + training loop
scripts/train_franka_reach.py # training CLI entry point
scripts/view_env.py           # interactive MuJoCo GUI viewer (+ policy rollout)
tests/test_smoke.py           # end-to-end shape / one-iteration tests
```

## Install

```bash
conda create -n vision_rl python=3.11 -y
conda activate vision_rl
pip install "jax[cuda12]"          # GPU build (needs an NVIDIA driver)
pip install flax optax mujoco mujoco-mjx
pip install -e .                    # make `vision_rl` importable everywhere
pip install wandb                   # optional: metric logging
```

pip install madrona-mjx    # or build from source
```

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6   # ~2.4 GB of a 4 GB card
export MUJOCO_GL=glfw                        # or: osmesa (headless), egl
```

MUJOCO_GL=glfw python scripts/view_env.py --task so101_pick_place --random

```bash
python scripts/train_franka_reach.py --task so101_pick_place --backend cpu --num-envs 16 --gui


Train command lines : 
```
# tile up to 16 worlds (default)
python -m vision_rl.train --task so101_pick_place --backend cpu --num-envs 16 --gui

# choose how many tiles to show
python -m vision_rl.train --task so101_pick_place --num-envs 16 --gui --gui-envs 9

# don't throttle to real-time (faster training, choppier view)
python -m vision_rl.train --task so101_pick_place --num-envs 16 --gui --gui-fast


`
