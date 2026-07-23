"""PPO training loop for vision-based SO-101 manipulation on MJX.

Puts the pieces together:
    VisionVecEnv (MJX physics + batch render + frame stack)
        -> ActorCritic (CNN encoder + Gaussian policy + value)
        -> rollout collection (jitted lax.scan)
        -> GAE -> PPO minibatch updates.

Run with `python -m vision_rl.train`.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from vision_rl import checkpoint as ckpt_io
from vision_rl.config import Config
from vision_rl.envs import VisionVecEnv
from vision_rl.models import ActorCritic
from vision_rl.ppo import compute_gae, create_train_state, ppo_update


def _init_wandb(cfg: Config, n_params: int):
    """Start a Weights & Biases run; config is logged for reproducibility."""
    import dataclasses

    import wandb

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name or cfg.exp_name,
        config={**dataclasses.asdict(cfg), "n_params": n_params},
    )
    print(f"[wandb] logging to {run.url}")
    return run


def _build_network(cfg: Config, action_size: int) -> ActorCritic:
    return ActorCritic(
        action_dim=action_size,
        encoder_channels=cfg.encoder.channels,
        encoder_kernels=cfg.encoder.kernel_sizes,
        encoder_strides=cfg.encoder.strides,
        encoder_features=cfg.encoder.features,
        normalize_pixels=cfg.encoder.normalize_pixels,
        init_log_std=cfg.ppo.init_log_std,
        log_std_min=cfg.ppo.log_std_min,
        log_std_max=cfg.ppo.log_std_max,
        layer_norm=cfg.encoder.layer_norm,
    )


def make_train(cfg: Config) -> Callable[[], dict]:
    env = VisionVecEnv(cfg)
    network = _build_network(cfg, env.action_size)
    ppo = cfg.ppo

    num_updates = ppo.total_env_steps // cfg.batch_size
    n_gui = min(ppo.num_envs, cfg.gui_envs)   # worlds mirrored in the tiled viewer

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
                "table_hit": next_vstate.env_state.metrics["table_hit"],
                # First n_gui worlds' state, for the optional tiled GUI mirror.
                "qpos": next_vstate.env_state.data.qpos[:n_gui],
                "mocap": next_vstate.env_state.data.mocap_pos[:n_gui],
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
            "table_hit_rate": traj["table_hit"].mean(),
        }
        # Rollout trajectory of the first n_gui worlds, for the tiled GUI (kept
        # on device until the driver pulls it; negligible cost when GUI is off).
        viz = {"qpos": traj["qpos"], "mocap": traj["mocap"]}
        return train_state, vstate, metrics, viz

    # ----------------------------------------------------------------- #
    # Driver
    # ----------------------------------------------------------------- #
    def train():
        rng = jax.random.PRNGKey(ppo.seed)
        rng, init_rng, reset_rng = jax.random.split(rng, 3)

        train_state = create_train_state(
            init_rng, apply_fn, params_init, ppo, num_updates
        )
        # Resume: restore params AND opt_state, so Adam's momentum and the LR
        # schedule's counter carry over. `start_it` makes the loop continue the
        # original iteration numbering rather than re-running the full budget.
        # `run_dir` groups a run's checkpoints in one timestamped folder; a
        # resume keeps writing into the folder it was resumed from, so a run
        # interrupted and restarted any number of times stays in one place.
        start_it = 0
        run_dir = os.path.join(
            cfg.ckpt_dir, f"{cfg.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}")
        if cfg.resume_ckpt:
            ck = ckpt_io.load(cfg.resume_ckpt)
            train_state = train_state.replace(
                params=ck.restore_params(train_state.params),
                opt_state=ck.restore_opt_state(train_state.opt_state),
            )
            start_it = ck.step // cfg.batch_size
            if start_it >= num_updates:
                raise SystemExit(
                    f"[resume] {os.path.basename(cfg.resume_ckpt)} is already at "
                    f"{ck.step:,} steps — the full budget it was launched with "
                    f"({ppo.total_env_steps:,}). Its LR schedule has run to "
                    f"completion, so extending it would mean re-annealing from "
                    f"scratch; start a fresh run with a larger --steps instead."
                )
            run_dir = os.path.dirname(os.path.abspath(cfg.resume_ckpt))
            print(f"[resume] {os.path.basename(cfg.resume_ckpt)} @ {ck.step:,} steps "
                  f"(iter {start_it}/{num_updates}); config restored from checkpoint")
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(train_state.params))
        print(f"[init] renderer backend = {env.renderer.backend}")
        if env.env.impl == "warp":
            base = env.env
            print(f"[init] warp buffers: naconmax = {base._naconmax:,} "
                  f"({base._naconmax // max(1, ppo.num_envs)}/world), "
                  f"njmax = {base._njmax:,}/world")
        print(f"[init] policy params = {n_params:,}")
        print(f"[init] updates = {num_updates}, batch = {cfg.batch_size}, "
              f"minibatch = {cfg.minibatch_size}")
        print(f"[init] checkpoints -> {run_dir}/")

        run = _init_wandb(cfg, n_params) if cfg.use_wandb else None
        gui = None
        if cfg.gui:
            from vision_rl.envs.tiled_viewer import TiledMirror
            gui = TiledMirror(env.xml_path, env.mj_model, n_show=n_gui,
                              realtime=cfg.gui_realtime, ctrl_dt=env.ctrl_dt)

        vstate = env.reset(reset_rng)

        os.makedirs(run_dir, exist_ok=True)
        history = []
        start = time.time()
        for it in range(start_it + 1, num_updates + 1):
            rng, it_rng = jax.random.split(rng)
            train_state, vstate, metrics, viz = train_iteration(
                train_state, vstate, it_rng
            )
            if gui is not None:
                gui.replay(viz)

            if it % ppo.log_interval == 0:
                metrics = jax.tree_util.tree_map(lambda x: float(x), metrics)
                history.append({"iter": it, **metrics})
                steps = it * cfg.batch_size                  # cumulative across resumes
                session_steps = (it - start_it) * cfg.batch_size
                sps = session_steps / (time.time() - start)  # this session's throughput
                print(
                    f"[{it:>5}/{num_updates}] steps={steps:>10,} "
                    f"R={metrics['mean_reward']:+.3f} "
                    f"dist={metrics['mean_dist']:.3f} "
                    f"succ={metrics['success_rate']:.2f} "
                    f"tbl={metrics['table_hit_rate']:.2f} "
                    f"kl={metrics['approx_kl']:.4f} "
                    f"ent={metrics['entropy']:+.2f} "
                    f"| {sps:,.0f} sps"
                )
                if run is not None:
                    run.log({**metrics, "sps": sps}, step=steps)

            # Checkpoint: by env-steps if ckpt_every_steps set, else by iterations.
            steps_done = it * cfg.batch_size
            if ppo.ckpt_every_steps > 0:
                save_now = (steps_done // ppo.ckpt_every_steps
                            > (steps_done - cfg.batch_size) // ppo.ckpt_every_steps)
                tag = f"step{steps_done}"
            else:
                save_now = (it % ppo.ckpt_interval == 0)
                tag = f"it{it}"
            if save_now or it == num_updates:
                path = os.path.join(run_dir, f"{cfg.exp_name}_{tag}.msgpack")
                ckpt_io.save(path, train_state.params, train_state.opt_state,
                             cfg, steps_done)
                print(f"[ckpt] saved {os.path.basename(path)}")

        if run is not None:
            run.finish()
        if gui is not None:
            gui.close()
        return {"train_state": train_state, "history": history}

    return train


def main():
    import argparse

    from vision_rl.config import small_config, so101_config, so101_pick_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="so101_pick_place",
                        choices=["so101_pick_place", "so101_pick"])
    parser.add_argument("--small", action="store_true", help="tiny config for tests")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--backend", type=str, default=None,
                        choices=["warp", "madrona", "cpu", "auto"])
    parser.add_argument("--res", type=int, default=None,
                        help="square render resolution (default 84 for so101)")
    parser.add_argument("--entropy-coef", type=float, default=None,
                        help="override PPO entropy bonus coefficient")
    parser.add_argument("--naconmax", type=int, default=None,
                        help="warp contact buffer, TOTAL across all worlds "
                             "(default: sized from the scene)")
    parser.add_argument("--njmax", type=int, default=None,
                        help="warp constraint rows PER world; must cover the "
                             "worst world (default: sized from the scene)")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--ckpt-steps", type=int, default=None,
                        help="save a checkpoint every N env-steps (e.g. 100000)")
    parser.add_argument("--resume", type=str, default=None,
                        help="continue a run from a checkpoint .msgpack: params, "
                             "optimizer state, step count and ALL training "
                             "parameters come from the file")
    parser.add_argument("--gui", action="store_true",
                        help="open a live window tiling training worlds")
    parser.add_argument("--gui-envs", type=int, default=None,
                        help="max worlds to show in the tiled GUI (default 16)")
    parser.add_argument("--gui-fast", action="store_true",
                        help="with --gui, replay as fast as possible (not real-time)")
    args = parser.parse_args()

    cfg = so101_pick_config() if args.task == "so101_pick" else so101_config()
    if args.small:
        cfg.ppo.num_envs = 32
        cfg.ppo.rollout_length = 8
        cfg.ppo.num_minibatches = 4
        cfg.ppo.total_env_steps = 200_000
        cfg.render.width = cfg.render.height = 48
    if args.num_envs is not None:
        cfg.ppo.num_envs = args.num_envs
    if args.steps is not None:
        cfg.ppo.total_env_steps = args.steps
    if args.backend is not None:
        cfg.render.backend = args.backend
    if args.res is not None:
        cfg.render.width = cfg.render.height = args.res
    if args.entropy_coef is not None:
        cfg.ppo.entropy_coef = args.entropy_coef
    if args.naconmax is not None:
        cfg.ppo.naconmax = args.naconmax
    if args.njmax is not None:
        cfg.ppo.njmax = args.njmax
    if args.wandb:
        cfg.use_wandb = True
    if args.wandb_project is not None:
        cfg.wandb_project = args.wandb_project
    if args.wandb_name is not None:
        cfg.wandb_run_name = args.wandb_name
    if args.ckpt_steps is not None:
        cfg.ppo.ckpt_every_steps = args.ckpt_steps
    if args.gui:
        cfg.gui = True
    if args.gui_envs is not None:
        cfg.gui_envs = args.gui_envs
    if args.gui_fast:
        cfg.gui_realtime = False

    if args.resume is not None:
        # A resume is a continuation, not a new run: every training parameter
        # comes from the checkpoint so the LR schedule, reward shaping and env
        # geometry stay exactly as they were. Only session choices (backend,
        # GUI, wandb, ckpt_dir) follow the current command line.
        ignored = [f"--{n.replace('_', '-')}" for n in (
            "task", "small", "num_envs", "steps", "res", "entropy_coef",
            "naconmax", "njmax", "ckpt_steps")
            if getattr(args, n) != parser.get_default(n)]
        cfg = ckpt_io.apply_session_fields(ckpt_io.load_config(args.resume), cfg)
        cfg.resume_ckpt = args.resume
        if ignored:
            print(f"[resume] ignoring {', '.join(ignored)} — training parameters "
                  f"come from the checkpoint")

    train = make_train(cfg)
    train()


if __name__ == "__main__":
    main()
