"""Configuration dataclasses for the vision-PPO pipeline.

All knobs live here so experiments are reproducible from a single object.
Defaults are tuned to fit a 4 GB GPU (small images, modest env count).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EnvConfig:
    """Franka reach environment settings."""

    episode_length: int = 200          # env steps per episode before truncation
    action_repeat: int = 1             # physics steps skipped between policy steps
    ctrl_dt: float = 0.02              # control timestep (s)
    sim_dt: float = 0.004              # physics timestep (s); n_substeps = ctrl/sim

    # Target sampling volume (metres, relative to the robot base).
    target_low: Tuple[float, float, float] = (0.30, -0.25, 0.15)
    target_high: Tuple[float, float, float] = (0.55, 0.25, 0.45)

    # Reward shaping.
    reach_reward_scale: float = 1.0    # weight on the -distance dense term
    success_threshold: float = 0.05    # metres; within this = success bonus
    success_bonus: float = 2.0
    ctrl_cost_scale: float = 0.01      # penalty on action magnitude

    action_scale: float = 0.5          # scales normalized [-1,1] action to ctrlrange


@dataclass
class SO101Config:
    """SO-101 pick-and-place environment settings."""

    episode_length: int = 250
    ctrl_dt: float = 0.02
    action_scale: float = 0.3          # residual joint-target scale (arm)

    # Cube spawn region (metres, world frame, in front of the arm along +x).
    cube_low: Tuple[float, float] = (0.13, -0.05)
    cube_high: Tuple[float, float] = (0.19, 0.05)
    cube_z: float = 0.015              # cube half-size = resting height
    # Place-target region (where the cube must be delivered).
    target_low: Tuple[float, float] = (0.13, 0.08)
    target_high: Tuple[float, float] = (0.19, 0.16)

    # Reward shaping (staged: reach -> lift -> place).
    reach_scale: float = 1.0
    lift_scale: float = 3.0
    lift_thresh: float = 0.06          # cube_z above which it counts as lifted
    max_lift: float = 0.08
    place_scale: float = 2.0
    place_radius: float = 0.15         # normalization radius for place reward
    success_thresh: float = 0.04       # xy distance to target for success
    success_bonus: float = 5.0
    ctrl_cost_scale: float = 0.01


@dataclass
class RenderConfig:
    """Batch camera rendering settings."""

    width: int = 64
    height: int = 64
    camera: str = "overhead_cam"
    backend: str = "auto"              # "madrona" | "cpu" | "auto"
    gpu_id: int = 0
    use_rasterizer: bool = False       # Madrona: rasterizer vs raytracer
    # Geom groups visible to the batch renderer (0,1,2 = default visual groups).
    enabled_geom_groups: Tuple[int, ...] = (0, 1, 2)


@dataclass
class EncoderConfig:
    """CNN vision encoder settings."""

    channels: Tuple[int, ...] = (32, 64, 64)   # conv channels per block
    kernel_sizes: Tuple[int, ...] = (8, 4, 3)
    strides: Tuple[int, ...] = (4, 2, 1)
    features: int = 256                        # embedding dim after the CNN
    frame_stack: int = 3                       # stacked frames -> temporal cue
    normalize_pixels: bool = True              # divide uint8 by 255 -> [0,1]


@dataclass
class PPOConfig:
    """PPO / optimisation hyper-parameters."""

    total_env_steps: int = 20_000_000
    num_envs: int = 256                # parallel MJX worlds (tune to VRAM)
    rollout_length: int = 16           # steps collected per env per iteration
    num_minibatches: int = 8
    update_epochs: int = 4

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    learning_rate: float = 3e-4
    anneal_lr: bool = True
    normalize_advantages: bool = True

    # Continuous-action policy: state-independent log-std, initialised here.
    init_log_std: float = -0.5

    seed: int = 0
    log_interval: int = 1              # iterations between console logs
    eval_interval: int = 50           # iterations between eval rollouts
    ckpt_interval: int = 100          # iterations between checkpoints


@dataclass
class Config:
    # Which task to build: "franka_reach" | "so101_pick_place".
    task: str = "franka_reach"
    env: EnvConfig = field(default_factory=EnvConfig)
    so101: SO101Config = field(default_factory=SO101Config)
    render: RenderConfig = field(default_factory=RenderConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    exp_name: str = "franka_reach_vision_ppo"
    ckpt_dir: str = "checkpoints"

    # Weights & Biases logging.
    use_wandb: bool = False
    wandb_project: str = "vision-rl"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

    # Live GUI: tile training worlds in one MuJoCo window.
    gui: bool = False
    gui_realtime: bool = True          # throttle replay to real-time (watchable)
    gui_envs: int = 16                 # max worlds to display (grid tiles)

    @property
    def batch_size(self) -> int:
        return self.ppo.num_envs * self.ppo.rollout_length

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.ppo.num_minibatches


def small_config() -> Config:
    """A tiny config for smoke tests / very small GPUs."""
    cfg = Config()
    cfg.ppo.num_envs = 32
    cfg.ppo.rollout_length = 8
    cfg.ppo.num_minibatches = 4
    cfg.ppo.total_env_steps = 200_000
    cfg.render.width = 48
    cfg.render.height = 48
    return cfg


def so101_config() -> Config:
    """Default config for the SO-101 pick-and-place task."""
    cfg = Config()
    cfg.task = "so101_pick_place"
    cfg.exp_name = "so101_pick_place_vision_ppo"
    cfg.ppo.num_envs = 256
    cfg.ppo.rollout_length = 32          # longer horizon for a multi-stage task
    cfg.ppo.num_minibatches = 8
    cfg.ppo.entropy_coef = 0.005
    return cfg
