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
    action_scale: float = 0.9           # residual joint-target range (rad) around
                                        # home; covers the spawn region + descent
                                        # (near corners need ~0.84 rad base yaw)

    # Cube spawn region (metres) — reduced to the SOLID top-down-graspable zone
    # confirmed by point-down IK reachability (menagerie SO-101).
    cube_low: Tuple[float, float] = (0.14, -0.11)
    cube_high: Tuple[float, float] = (0.22, 0.11)
    cube_z: float = 0.015              # cube half-size = resting height (3 cm cube)
    # Place-target region (sampled independently -> varied carry distances).
    target_low: Tuple[float, float] = (0.14, -0.11)
    target_high: Tuple[float, float] = (0.22, 0.11)

    # PHYSICAL grasping (no assist): the gripper must really close on the cube.
    # `grasp_dist` gates the grasp-shaping bonus (fingers at the cube).
    grasp_dist: float = 0.045

    # Reward shaping (staged: reach -> grasp -> lift -> place). Kept modest so the
    # return scale stays stable (the old scale=10 blew up the value loss).
    reach_scale: float = 1.0
    grasp_bonus_scale: float = 0.5     # reward closing the jaw when at the cube
    lift_scale: float = 4.0            # the real grasp signal (cube only rises if gripped)
    lift_thresh: float = 0.05          # cube_z above which it counts as lifted (3.5cm off table)
    max_lift: float = 0.10
    place_scale: float = 2.0
    place_radius: float = 0.15         # normalization radius for place reward
    success_thresh: float = 0.04       # xy distance to target for success
    success_bonus: float = 5.0
    ctrl_cost_scale: float = 0.01
    # Penalty per step while the gripper dives into the table away from the cube.
    table_penalty_scale: float = 0.5

    # Dense progress shaping: reward per-step *improvement* toward the goal.
    reach_progress_scale: float = 3.0    # gripper approaching the cube
    place_progress_scale: float = 3.0    # lifted cube approaching the pad


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
    layer_norm: bool = True                    # LayerNorm before each tanh/relu;
                                               # without it the trunk's tanh
                                               # saturates and the CNN gets no
                                               # gradient (see models/encoder.py)


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
    # Anneal the LR DOWN to this fraction of the initial rate (not to 0), so
    # late-training discoveries can still be reinforced. Fully-zero LR froze
    # past runs right when success first appeared (~75-85% through training).
    lr_final_frac: float = 0.30
    normalize_advantages: bool = True

    # Continuous-action policy: state-independent log-std, initialised here and
    # hard-clamped to [log_std_min, log_std_max] so noise can't collapse or run
    # away (sigma in [0.05, 1.35]).
    init_log_std: float = -0.5
    log_std_min: float = -3.0
    log_std_max: float = 0.3

    # Warp physics buffers; None = size automatically from the scene (see
    # envs/buffers.py). Set these only to override that sizing.
    #   naconmax: GLOBAL contact buffer, summed over all worlds.
    #   njmax:    PER-WORLD constraint rows; must cover the WORST world, which
    #             is far above the average (rare gripper/cube/table pile-ups).
    # Too small = warp silently drops contacts and the physics goes wrong.
    naconmax: int | None = None
    njmax: int | None = None

    seed: int = 0
    log_interval: int = 1              # iterations between console logs
    eval_interval: int = 50           # iterations between eval rollouts
    ckpt_interval: int = 1000          # iterations between checkpoints (fallback)
    ckpt_every_steps: int = 0          # env-steps between checkpoints (0 = use ckpt_interval)


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
    resume_ckpt: str | None = None     # path to a params .msgpack to warm-start from

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
    # Entropy bonus: 0.001 was too weak — exploration collapsed (entropy -> -6.8,
    # sigma near its floor) by mid-training, before the grasp->carry->place skill
    # could consolidate. 0.02 keeps exploration alive deeper into training while
    # still letting the policy sharpen late for precise (4 cm) placement.
    cfg.ppo.entropy_coef = 0.02
    # 84x84 -> 7x7x64 flatten, the shape the (8,4,3)/(4,2,1) stack was designed
    # for. Resolves the 2 cm cube in ~6 px; 224 only inflates the encoder Dense.
    cfg.render.width = cfg.render.height = 84
    return cfg
