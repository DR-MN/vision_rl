from vision_rl.envs.franka_reach import FrankaReachEnv, EnvState
from vision_rl.envs.so101_pick_place import SO101PickPlaceEnv
from vision_rl.envs.vision_wrapper import VisionVecEnv


def make_env(cfg):
    """Build the base (single-world) env selected by cfg.task."""
    task = getattr(cfg, "task", "franka_reach")
    if task == "franka_reach":
        return FrankaReachEnv(cfg.env)
    if task == "so101_pick_place":
        return SO101PickPlaceEnv(cfg.so101)
    raise ValueError(f"unknown task: {task}")


__all__ = [
    "FrankaReachEnv", "SO101PickPlaceEnv", "EnvState", "VisionVecEnv", "make_env",
]
