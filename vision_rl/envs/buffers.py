"""Sizing for the Warp physics buffers (`naconmax` / `njmax`).

MuJoCo-Warp allocates contact/constraint storage once, up front, with static
shapes baked into the compiled kernels — they cannot grow at runtime. If a
buffer is too small the solver **silently drops** the excess and only prints a
warning, so the sim keeps running on wrong physics (arm sinks through the
table, grasps fail). Sizing therefore has to be right before the first step.

The two buffers are scoped differently, which is easy to get wrong:

  naconmax  GLOBAL — total contacts summed over every world. A sum over many
            worlds is stable, so an average-per-world budget + headroom is the
            correct way to size it.
  njmax     PER-WORLD — must cover the single WORST world. Most worlds have the
            arm in free space with ~7 rows; a rare gripper/cube/table pile-up
            needs 20x that. Sizing this off the average is what overflows.

Sizing for the theoretical maximum is not an option (this scene admits ~1951
contacts / ~10k rows per world = hundreds of MB per 1k worlds). Instead we
probe the actual model and apply headroom.

Note the probe drives random actions, which *under*-reports what a trained
policy does: a random policy rarely touches anything, while a learning policy
presses into the table and grips the cube, and the requirement keeps climbing
as it improves. Measured here: random peaks at ~64 rows where a mid-training
policy already needed 142. Hence the deliberately large safety factors and
floors below — the probe sets a lower bound, not the answer.
"""

from __future__ import annotations

import numpy as np
import mujoco


# Probe peaks are a lower bound on trained-policy demand (see module docstring),
# so multiply generously. Memory is cheap here: even 4k worlds at these sizes is
# a few hundred MB, versus silently-corrupted physics if we come up short.
_CONTACT_SAFETY = 4.0
_EFC_SAFETY = 4.0

# Floors, so a lucky (low-contact) probe can never size us below what a trained
# policy is already known to need.
_MIN_CONTACTS_PER_WORLD = 32
_MIN_NJMAX = 256

# Ceiling on njmax, purely to catch a pathological scene before it eats the GPU.
_MAX_NJMAX = 4096


def probe_peaks(mj_model: mujoco.MjModel, steps: int = 3000,
                seed: int = 0) -> tuple[int, int]:
    """Run random actions on CPU MuJoCo and return peak (ncon, nefc) per world.

    Cheap (~1s) and runs before any GPU allocation. Returns observed peaks; the
    caller is responsible for adding headroom.
    """
    d = mujoco.MjData(mj_model)
    rng = np.random.default_rng(seed)
    lo, hi = mj_model.actuator_ctrlrange[:, 0], mj_model.actuator_ctrlrange[:, 1]

    has_key = mj_model.nkey > 0
    reset = (lambda: mujoco.mj_resetDataKeyframe(mj_model, d, 0)) if has_key \
        else (lambda: mujoco.mj_resetData(mj_model, d))
    reset()

    peak_con = peak_efc = 0
    for i in range(steps):
        # Full-range targets drive the arm into the table/cube rather than
        # hovering, which is where the contact pile-ups actually happen.
        d.ctrl[:] = rng.uniform(lo, hi)
        for _ in range(10):
            mujoco.mj_step(mj_model, d)
        peak_con = max(peak_con, int(d.ncon))
        peak_efc = max(peak_efc, int(d.nefc))
        if i % 500 == 0:
            reset()

    return peak_con, peak_efc


def size_buffers(mj_model: mujoco.MjModel, num_envs: int,
                 naconmax: int | None = None, njmax: int | None = None,
                 probe: bool = True) -> tuple[int, int]:
    """Return (naconmax, njmax) for `num_envs` worlds of `mj_model`.

    Explicit values pass through untouched; anything left as None is derived
    from a probe of this model (or from the floors, if `probe` is False).
    """
    if naconmax is not None and njmax is not None:
        return naconmax, njmax

    per_world_contacts, efc = _MIN_CONTACTS_PER_WORLD, _MIN_NJMAX
    if probe:
        peak_con, peak_efc = probe_peaks(mj_model)
        per_world_contacts = max(per_world_contacts,
                                 int(np.ceil(peak_con * _CONTACT_SAFETY)))
        efc = max(efc, int(np.ceil(peak_efc * _EFC_SAFETY)))

    if naconmax is None:
        naconmax = per_world_contacts * num_envs
    if njmax is None:
        njmax = min(efc, _MAX_NJMAX)

    return int(naconmax), int(njmax)
