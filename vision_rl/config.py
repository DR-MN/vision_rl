"""Configuration dataclasses for the vision-PPO pipeline.

All knobs live here so experiments are reproducible from a single object.
Defaults are tuned to fit a 4 GB GPU (small images, modest env count).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SO101Config:
    """SO-101 pick-and-place environment settings."""

    episode_length: int = 250
    ctrl_dt: float = 0.02
    action_scale: float = 0.9           # residual joint-target range (rad) around
                                        # home; covers the spawn region + descent
                                        # (near corners need ~0.84 rad base yaw)
    # Per-control-step slew limit on arm joint targets (rad); caps commanded
    # angular velocity so the arm doesn't snap to a new target in one 20ms
    # step. Mirrors the real-deploy slew limit (scripts/run_real.py --max-step-rad).
    max_joint_delta: float = 0.08

    # Cube spawn region (metres) — the SOLID top-down-graspable zone confirmed
    # by point-down IK reachability (menagerie SO-101), pushed out in x a bit
    # so the arm has to extend rather than staying folded near the base.
    cube_low: Tuple[float, float] = (0.14, -0.11)
    cube_high: Tuple[float, float] = (0.25, 0.11)
    cube_z: float = 0.0175              # cube half-height = resting height (3.5 cm cube)
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
    lift_thresh: float = 0.0525        # cube_z above which it counts as lifted (3.5cm off table)
    max_lift: float = 0.10
    place_scale: float = 2.0
    place_radius: float = 0.15         # normalization radius for place reward
    success_thresh: float = 0.04       # xy distance to target for success
    success_bonus: float = 5.0
    ctrl_cost_scale: float = 0.01
    # Penalty per step while the gripper dives into the table away from the cube.
    table_penalty_scale: float = 0.5
    # Virtual clearance plane above the table top (z=0): the gripper dipping
    # below this height while away from the cube counts as a table hit.
    table_clear: float = 0.02

    # Dense progress shaping: reward per-step *improvement* toward the goal.
    reach_progress_scale: float = 3.0    # gripper approaching the cube
    place_progress_scale: float = 3.0    # lifted cube approaching the pad


@dataclass
class SO101PickConfig:
    """SO-101 pick-only environment settings (grasp + lift, no place)."""

    episode_length: int = 250
    ctrl_dt: float = 0.03
    action_scale: float = 1.8
    # Per-control-step slew limit on arm joint targets (rad); see SO101Config.
    # Keeps the arm slow even though action_scale is large.
    max_joint_delta: float = 0.08

    # Cube spawn region (metres); see SO101Config.cube_low/high.
    cube_low: Tuple[float, float] = (0.15, -0.13)
    cube_high: Tuple[float, float] = (0.27, 0.13)
    cube_z: float = 0.0175              # cube half-height = resting height (3.5 cm cube)

    grasp_dist: float = 0.045          # horizontal radius gating the table penalty

    # --- TOP-DOWN approach shaping -------------------------------------- #
    # The pick is staged as: get the gripper horizontally OVER the cube, then
    # descend onto it. Reach is therefore split into a horizontal component
    # (align) and a vertical one (descend), and the descent reward only pays
    # while aligned -- so a sideways swipe into the cube earns nothing.
    align_tol: float = 0.02            # horizontal offset counted as "over the cube"
    grasp_z_tol: float = 0.03          # |gripper_z - cube_z| allowed for a grasp
    align_scale: float = 1.0           # pull the gripper over the cube (horizontal)
    align_progress_scale: float = 3.0  # per-step horizontal progress
    # Continuous cost for however high above the grasp band the gripper
    # currently is, once aligned -- the vertical mirror of `align_scale`.
    # Added 2026-08-06: `descend_progress_scale` alone only pays for a NEW
    # record-low `dz` (see the min_dz ratchet in so101_pick.py, BUG D fix),
    # which made "aligned but never descending" exactly reward-NEUTRAL in
    # every direction, including up -- a trained policy aligned perfectly
    # (d_xy converged to ~1cm) but then drifted from dz~0.24 up to dz~0.45
    # and parked there for the rest of the episode, because nothing was
    # pulling it back down once the ratchet stopped paying. `align_r` never
    # has this problem because `d_xy` carries a continuous cost AND a
    # progress bonus; `dz` only had the latter. `descend_scale` gives it the
    # former too. NOT gated on `grip_open` (unlike `descend_progress_scale`)
    # so it can't be dodged the way BUG D's exploit dodged the debit for
    # ascending -- it applies regardless of jaw state.
    descend_scale: float = 1.0
    # Per-step descent progress. Gated on aligned AND on the jaw being OPEN --
    # see open_bonus_scale below for why the open gate is there.
    descend_progress_scale: float = 3.0

    # Reward shaping (staged: align -> OPEN -> descend -> grasp -> lift).
    # ONE-TIME bonus (latched via `was_open`, see so101_pick.py step()) for
    # first opening the jaw while aligned over the cube but still above it.
    # Without ANY open incentive, every gripper term rewards CLOSING
    # (grasp_bonus, the `held` precondition on success) and nothing rewards
    # opening, so a trained policy keeps the jaw shut permanently, approaches
    # correctly, and just bumps the cube -- a closed jaw cannot enclose the
    # 4.5cm cube, so grasp/lift/success never fire and there is no gradient
    # out of it. This USED to pay open_bonus_scale every step the pose held
    # (2026-08-05: found via a 50M-step run that plateaued at R~0.47 for 30M+
    # steps with success=0.00) -- that flat per-step version let the policy
    # camp aligned-and-open indefinitely (any dz > grasp_z_tol qualifies) and
    # out-earn actually finishing the pick, since descend_pr's whole-episode
    # total is a few tenths, dwarfed by N steps of a standing 0.5 bonus. Now
    # it only ever fires once per episode, so it can bootstrap "open before
    # descending" but never fund parking there.
    open_bonus_scale: float = 0.5
    grasp_bonus_scale: float = 0.5
    lift_scale: float = 4.0
    lift_thresh: float = 0.0525        # cube_z above which it counts as lifted (3.5cm off table)
    pick_height: float = 0.10          # cube must rise this high above the table for a successful pick
    success_bonus: float = 5.0
    ctrl_cost_scale: float = 0.01
    table_penalty_scale: float = 0.5
    # Virtual clearance plane above the table top (z=0); see SO101Config.table_clear.
    table_clear: float = 0.02


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
    # Which task to build: "so101_pick_place" | "so101_pick".
    task: str = "so101_pick_place"
    so101: SO101Config = field(default_factory=SO101Config)
    pick: SO101PickConfig = field(default_factory=SO101PickConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    exp_name: str = "so101_pick_place_vision_ppo"
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
    gui_envs: int = 5               # max worlds to display (grid tiles)

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
    cfg.ppo.entropy_coef = 0.005
    # 84x84 -> 7x7x64 flatten, the shape the (8,4,3)/(4,2,1) stack was designed
    # for. Resolves the 2 cm cube in ~6 px; 224 only inflates the encoder Dense.
    cfg.render.width = cfg.render.height = 84
    return cfg


def so101_pick_config() -> Config:
    """Default config for the SO-101 pick-only task (grasp + lift, no place)."""
    cfg = Config()
    cfg.task = "so101_pick"
    cfg.exp_name = "so101_pick_vision_ppo"
    cfg.ppo.num_envs = 256
    cfg.ppo.rollout_length = 32
    cfg.ppo.num_minibatches = 8
    cfg.ppo.entropy_coef = 0.005
    cfg.render.width = cfg.render.height = 84
    return cfg
