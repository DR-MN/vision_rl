from vision_rl.envs.buffers import probe_peaks, size_buffers
from vision_rl.envs.franka_reach import FrankaReachEnv, EnvState
from vision_rl.envs.so101_pick_place import SO101PickPlaceEnv
from vision_rl.envs.so101_pick import SO101PickEnv
from vision_rl.envs.vision_wrapper import VisionVecEnv


def make_env(cfg):
    """Build the base (single-world) env selected by cfg.task.

    Physics impl follows the render backend: the Warp renderer needs warp-impl
    physics, so `--backend warp` builds the env with `impl='warp'`.

    The warp contact/constraint buffers are sized per-scene (see envs.buffers);
    cfg.ppo.naconmax / cfg.ppo.njmax override the automatic sizing.
    """
    task = getattr(cfg, "task", "franka_reach")
    impl = "warp" if getattr(cfg.render, "backend", "") == "warp" else "jax"
    kwargs = dict(impl=impl, num_envs=cfg.ppo.num_envs,
                  naconmax=cfg.ppo.naconmax, njmax=cfg.ppo.njmax)
    if task == "franka_reach":
        return FrankaReachEnv(cfg.env, **kwargs)
    if task == "so101_pick_place":
        return SO101PickPlaceEnv(cfg.so101, **kwargs)
    if task == "so101_pick":
        return SO101PickEnv(cfg.pick, **kwargs)
    raise ValueError(f"unknown task: {task}")


__all__ = [
    "FrankaReachEnv", "SO101PickPlaceEnv", "SO101PickEnv", "EnvState",
    "VisionVecEnv", "make_env", "probe_peaks", "size_buffers",
]
