#!/usr/bin/env python
"""Print a policy's exploration noise (Gaussian sigma) from a checkpoint.

is my policy still exploring a lot, or has it sharpened up
sigma < 0.25 = sharp/converged, 0.25–0.5 = normal mid-training, > 0.5 = noise is high enough

The policy is a diagonal Gaussian with a state-independent, learnable `log_std`
(one value per action dim). Its exploration noise is sigma = exp(log_std), and
the differential entropy the training log reports is

    entropy = 0.5 * action_dim * (1 + log(2*pi)) + sum(log_std)
            = action_dim * (1.4189385) + sum(log_std)

so a rising `ent` in the log means sigma is growing -- the policy is getting
noisier, not sharper. This reads log_std straight out of the params and reports
sigma per action dim (in the normalized [-1,1] action space) and, via
so101.action_scale, the actual per-joint jitter in radians.

No physics/render needed -- it only unpacks the msgpack, so it's instant and
runs in any env with flax installed.

Example:
    python scripts/check_std.py checkpoints/so101_pick_place_vision_ppo_step46412992.msgpack
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flax import serialization

from vision_rl.config import so101_config


def _find_log_std(tree):
    """Locate the 'log_std' leaf anywhere in a restored params pytree."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == "log_std":
                return v
            found = _find_log_std(v)
            if found is not None:
                return found
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to a params .msgpack")
    ap.add_argument("--action-scale", type=float, default=None,
                    help="rad per unit action (default: so101_config value)")
    args = ap.parse_args()

    with open(args.ckpt, "rb") as f:
        raw = serialization.msgpack_restore(f.read())
    log_std = _find_log_std(raw)
    if log_std is None:
        sys.exit("no 'log_std' parameter found in this checkpoint")
    log_std = np.asarray(log_std, dtype=np.float64)

    sigma = np.exp(log_std)
    action_dim = log_std.shape[0]
    entropy = action_dim * 0.5 * (1.0 + math.log(2 * math.pi)) + log_std.sum()
    scale = args.action_scale
    if scale is None:
        scale = so101_config().so101.action_scale

    print(f"\ncheckpoint : {os.path.basename(args.ckpt)}")
    print(f"action_dim : {action_dim}")
    print(f"entropy    : {entropy:+.3f}   (this is the 'ent' in the training log)")
    print(f"mean sigma : {sigma.mean():.4f}   (normalized [-1,1] action units)")
    print(f"           = {sigma.mean() * scale:.4f} rad jitter/step "
          f"= ±{math.degrees(sigma.mean() * scale):.1f}deg  (action_scale={scale})")
    print("\nper action dim:")
    for i, (ls, s) in enumerate(zip(log_std, sigma)):
        name = f"joint{i}" if i < action_dim - 1 else "gripper"
        print(f"  action[{i}] ({name:<7}) log_std={ls:+.3f}  sigma={s:.4f}  "
              f"±{math.degrees(s * scale):.1f}deg")

    print("\ninterpretation:")
    if sigma.mean() < 0.25:
        print("  sharp policy — low exploration noise, good for a converged run.")
    elif sigma.mean() < 0.5:
        print("  moderate noise — normal mid-training.")
    else:
        print("  HIGH noise — sigma>0.5 means the deterministic pi.mode() is far")
        print("  better than the stochastic rollout; entropy_coef is likely too high.")


if __name__ == "__main__":
    main()
