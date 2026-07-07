from vision_rl.envs.franka_reach import FrankaReachEnv, EnvState
from vision_rl.envs.so101_pick_place import SO101PickPlaceEnv
from vision_rl.envs.vision_wrapper import VisionVecEnv


def make_env(cfg):
    """Build the base (single-world) env selected by cfg.task.

    Physics impl follows the render backend: the Warp renderer needs warp-impl
    physics, so `--backend warp` builds the env with `impl='warp'`.
    """
    task = getattr(cfg, "task", "franka_reach")
    impl = "warp" if getattr(cfg.render, "backend", "") == "warp" else "jax"
    # Warp packs contacts into one global buffer sized ~num_envs * contacts/world;
    # give ~6/world of headroom so narrowphase never overflows.
    naconmax = max(64, 6 * cfg.ppo.num_envs)
    if task == "franka_reach":
        return FrankaReachEnv(cfg.env, impl=impl, naconmax=naconmax)
    if task == "so101_pick_place":
        return SO101PickPlaceEnv(cfg.so101, impl=impl, naconmax=naconmax)
    raise ValueError(f"unknown task: {task}")


__all__ = [
    "FrankaReachEnv", "SO101PickPlaceEnv", "EnvState", "VisionVecEnv", "make_env",
]
