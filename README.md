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
python -m vision_rl.train --task so101_pick_place --backend warp --num-envs 10 --gui

# can run with resolution and other parameters
python -m vision_rl.train --task so101_pick_place --backend warp --res 84 --num-envs 350 --steps 50000000 --ckpt-steps 200000 --wandb --gui
```

### Checkpoints and resuming

A checkpoint is one self-contained `.msgpack` holding the network weights, the
optimizer state (Adam moments + LR-schedule counter), the cumulative step count,
and the full `Config` that produced it.

Each run writes its checkpoints into its own timestamped folder, so runs never
mix and the launch time is visible from the directory listing:

```
checkpoints/
  so101_pick_place_vision_ppo_20260720-173949/
      so101_pick_place_vision_ppo_step20224.msgpack
      so101_pick_place_vision_ppo_step40192.msgpack
```

Resuming keeps writing into the folder the checkpoint came from, so a run
interrupted and restarted any number of times stays in one place.

Because the config travels with the weights, resuming needs nothing but the file
— task, resolution, reward shaping and step budget all come from it, so the run
continues under identical conditions instead of restarting the LR schedule:

```bash
python -m vision_rl.train --backend warp \
    --resume checkpoints/so101_pick_place_vision_ppo_step200000.msgpack
```

Only session choices (`--backend`, `--gui`, `--wandb`, `ckpt_dir`) follow the new
command line; training flags passed alongside `--resume` are ignored with a
warning. Eval and viewer scripts read the config from the checkpoint too, so
`--task`/`--res` are no longer needed there:

```bash
python scripts/eval.py --backend warp --num-envs 64 --ckpt <ckpt>
python scripts/check_vision.py --backend warp --ckpt <ckpt>
```
