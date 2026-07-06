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

## Why these choices

- **Vision-dependent by design.** The target position is *not* in the proprio
  vector — the agent must find it in the image. Proprio is only joint pos/vel +
  gripper xyz.
- **Residual position control.** Actions are deltas around a home pose, clipped
  to actuator ranges — smooth and stable for RL.
- **4 GB-GPU friendly defaults.** 64×64 images, small Nature-CNN encoder,
  moderate `num_envs`. Tune `num_envs` / resolution to your VRAM.

## Project layout

```
vision_rl/
├── config.py                 # all hyper-parameters (dataclasses)
├── envs/
│   ├── franka_reach.py       # single-world MJX env (physics + reward)
│   ├── vision_wrapper.py     # vmap + batch render + frame stack + auto-reset
│   └── assets/franka_emika_panda/
│       └── franka_reach.xml  # scene: Panda + mocap target + camera
├── rendering/
│   └── batch_renderer.py     # Madrona (GPU) and CPU rendering backends
├── models/
│   ├── encoder.py            # CNN vision encoder
│   └── networks.py           # actor-critic (Gaussian policy + value)
├── ppo/
│   ├── gae.py                # generalised advantage estimation
│   └── ppo.py                # train state, clipped loss, minibatch update
└── train.py                  # rollout collection + training loop
scripts/train_franka_reach.py # CLI entry point
tests/test_smoke.py           # end-to-end shape / one-iteration tests
```

## Install

```bash
conda create -n vision_rl python=3.11 -y
conda activate vision_rl
pip install "jax[cuda12]"          # GPU build (needs an NVIDIA driver)
pip install flax optax distrax mujoco mujoco-mjx
```

Optional (fast on-GPU rendering). The CPU renderer works out of the box; for
high-throughput training install the Madrona batch renderer:

```bash
# Requires CUDA toolkit + CMake; builds native code. See upstream for details:
#   https://github.com/shacklettbp/madrona_mjx
pip install madrona-mjx    # or build from source
```

Rendering backend is selected by `config.RenderConfig.backend`
(`"auto"` → Madrona if importable, else CPU) or the `--backend` CLI flag.

## Run

On a small GPU, cap JAX's memory and pick a working MuJoCo GL backend for the
CPU renderer (`egl` may be broken on some drivers; `glfw` needs a display,
`osmesa` is headless software rendering):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6   # ~2.4 GB of a 4 GB card
export MUJOCO_GL=glfw                        # or: osmesa (headless), egl
```

```bash
# Quick end-to-end check (CPU renderer, tiny config):
python -m tests.test_smoke

# Tiny training run:
python scripts/train_franka_reach.py --small --backend cpu

# Full run (uses Madrona if available):
python scripts/train_franka_reach.py

# Custom:
python scripts/train_franka_reach.py --num-envs 128 --steps 5000000 --backend madrona
```

Checkpoints (policy params) are written to `checkpoints/` as msgpack.

## Key hyper-parameters (`vision_rl/config.py`)

| Group    | Param                | Default | Notes                                  |
|----------|----------------------|---------|----------------------------------------|
| render   | `width`/`height`     | 64      | Lower to save VRAM                     |
| encoder  | `frame_stack`        | 3       | Temporal info for the CNN              |
| encoder  | `features`           | 256     | Embedding size after the conv stack    |
| ppo      | `num_envs`           | 256     | Parallel MJX worlds — main VRAM knob   |
| ppo      | `rollout_length`     | 16      | Steps per env per iteration            |
| ppo      | `clip_eps`           | 0.2     | PPO ratio clip                         |
| ppo      | `gamma`/`gae_lambda` | 0.99/0.95 | Discount / GAE                       |

## Notes & limits

- On a 4 GB GPU, start around `--num-envs 64` at 64×64 and scale up while
  watching `nvidia-smi`.
- The CPU renderer is correct but slow (host callback per step); use it for
  debugging, Madrona for real training throughput.
```
