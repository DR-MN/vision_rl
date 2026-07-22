"""Sim-to-real deployment for the SO-101 pick-and-place vision policy.

Modules:
    bridge  -- SO101Bridge: model action <-> joint targets, joints+camera -> obs
    policy  -- RealPolicy: batch-1, jitted, deterministic inference from a ckpt
    robots  -- RobotBackend + SimRobot (loopback) + LeRobotSO101 (hardware)

Run it with scripts/run_real.py (defaults to the hardware-free sim loopback).
"""

from vision_rl.deploy.bridge import SO101Bridge, JOINT_NAMES
from vision_rl.deploy.policy import RealPolicy
from vision_rl.deploy.robots import RobotBackend, SimRobot, LeRobotSO101

__all__ = [
    "SO101Bridge", "JOINT_NAMES", "RealPolicy",
    "RobotBackend", "SimRobot", "LeRobotSO101",
]
