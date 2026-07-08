# Vision-based RL with PPO on MuJoCo-JAX

## Install

```bash
conda create -n vision_rl python=3.11 -y
conda activate vision_rl
pip install "jax[cuda12]"          # GPU build (needs an NVIDIA driver)
pip install flax optax mujoco mujoco-mjx
pip install -e .                    # make `vision_rl` importable everywhere
pip install wandb                   # optional: metric logging
```

```bash
#CPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false  #tells jax not to allocate gpu space initally.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6   # ~2.4 GB of a 4 GB card
export MUJOCO_GL=glfw                        # or: osmesa (headless), egl
```
```bash
#train with gui shown , all env is showed, if needed to run headless, remove ( --gui )
python -m vision_rl.train --task so101_pick_place --backend cpu --num-envs 16 --gui
# if need to reduce the gui shown in gui use --gui-envs flag
python -m vision_rl.train --task so101_pick_place --backend cpu --num-envs 5 --gui --gui-envs 9

```

## GPU batch rendering (Warp) — `--backend warp`

CPU rendering is slow (host callback per step). MJX's built-in **Warp** GPU
renderer runs physics + rendering on-device (~6–10× faster)

```bash
pip install "warp-lang==1.13.0"

# no MUJOCO_GL needed (GPU raytracer, not GL)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
python -m vision_rl.train --task so101_pick_place --backend warp --num-envs 10 --gui

```
