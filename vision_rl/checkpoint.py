"""Self-contained training checkpoints.

A checkpoint is ONE msgpack file holding everything needed to reconstruct a run:

    {"format": 2,
     "step":   cumulative env steps completed,
     "config": the full Config that produced these weights,
     "params": policy + value network weights,
     "opt_state": Adam moments AND the LR-schedule counter}

Why the config lives inside the file: without it, loading a checkpoint means
guessing the settings it was trained under. The render resolution used to be
recovered by solving backwards from the encoder's Dense width -- which is
ambiguous (a band of resolutions share one conv output size), and which said
nothing at all about reward shaping, episode length or action scale. Those were
silently taken from whatever config.py happened to say at load time, so a
checkpoint became unreproducible the moment the reward function was retuned.

Why opt_state is saved: it is what makes `--resume` a true continuation. Adam's
momentum and, more importantly, the linear LR schedule's step counter live here.
Restoring only `params` restarts the schedule at the full learning rate, which
jolts a policy that was being carefully polished at 30% rate.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from flax import serialization

from vision_rl.config import Config

FORMAT_VERSION = 2

# Fields describing THIS invocation rather than the trained policy. On resume
# these keep the values from the current command line; everything else is taken
# from the checkpoint so training continues under identical conditions.
SESSION_FIELDS = frozenset({
    "ckpt_dir", "resume_ckpt",
    "use_wandb", "wandb_project", "wandb_entity", "wandb_run_name",
    "gui", "gui_realtime", "gui_envs",
    "render.backend", "render.gpu_id",
})


# --------------------------------------------------------------------- #
# Config <-> plain dict
# --------------------------------------------------------------------- #
def _untuple(x: Any) -> Any:
    """Tuples -> lists. flax packs msgpack with strict_types=True, which refuses
    tuples outright; `_restore_dataclass` converts them back on load."""
    if isinstance(x, (tuple, list)):
        return [_untuple(v) for v in x]
    if isinstance(x, dict):
        return {k: _untuple(v) for k, v in x.items()}
    return x


def config_to_dict(cfg: Config) -> dict:
    return _untuple(dataclasses.asdict(cfg))


def _restore_dataclass(target: Any, saved: dict) -> None:
    """Overwrite `target`'s fields in place from `saved`.

    msgpack has no tuple type, so tuple-valued fields (encoder channels, geom
    groups, sampling bounds) come back as lists. The live default tells us which
    fields must be coerced back -- more robust than introspecting annotations,
    which are strings here under `from __future__ import annotations`.
    """
    for f in dataclasses.fields(target):
        if f.name not in saved:
            continue                      # field added since; keep the default
        cur, new = getattr(target, f.name), saved[f.name]
        if dataclasses.is_dataclass(cur):
            _restore_dataclass(cur, new)
        elif isinstance(cur, tuple) and isinstance(new, list):
            setattr(target, f.name, tuple(new))
        else:
            setattr(target, f.name, new)


def config_from_dict(saved: dict) -> Config:
    cfg = Config()
    _restore_dataclass(cfg, saved)
    return cfg


def apply_session_fields(restored: Config, live: Config) -> Config:
    """Copy the current invocation's session choices onto a restored config."""
    for path in SESSION_FIELDS:
        obj_r, obj_l = restored, live
        *parents, leaf = path.split(".")
        for p in parents:
            obj_r, obj_l = getattr(obj_r, p), getattr(obj_l, p)
        setattr(obj_r, leaf, getattr(obj_l, leaf))
    return restored


# --------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------- #
def save(path: str, params, opt_state, cfg: Config, step: int) -> None:
    """Write a checkpoint atomically (a crash mid-write can't corrupt it)."""
    blob = serialization.msgpack_serialize({
        "format": FORMAT_VERSION,
        "step": int(step),
        "config": config_to_dict(cfg),
        "params": serialization.to_state_dict(params),
        "opt_state": serialization.to_state_dict(opt_state),
    })
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@dataclasses.dataclass
class Checkpoint:
    """A loaded checkpoint. `params`/`opt_state` are raw state dicts; use
    `restore_params` / `restore_opt_state` to graft them onto live pytrees."""

    config: Config
    step: int
    params: dict
    opt_state: dict

    def restore_params(self, target):
        return serialization.from_state_dict(target, self.params)

    def restore_opt_state(self, target):
        return serialization.from_state_dict(target, self.opt_state)


def load(path: str) -> Checkpoint:
    with open(path, "rb") as f:
        raw = serialization.msgpack_restore(f.read())

    fmt = raw.get("format")
    if fmt != FORMAT_VERSION:
        raise ValueError(
            f"{os.path.basename(path)}: unsupported checkpoint format {fmt!r} "
            f"(expected {FORMAT_VERSION}). Checkpoints written before the "
            f"self-contained format carried no config and cannot be loaded."
        )

    return Checkpoint(
        config=config_from_dict(raw["config"]),
        step=int(raw["step"]),
        params=raw["params"],
        opt_state=raw["opt_state"],
    )


def load_config(path: str) -> Config:
    """Just the config -- for eval/viewer scripts that rebuild the env first."""
    return load(path).config
