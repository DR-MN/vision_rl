#!/usr/bin/env python
"""Entry point: train vision PPO on Franka reach.

Examples:
    python scripts/train_franka_reach.py                # full run
    python scripts/train_franka_reach.py --small        # tiny (smoke) run
    python scripts/train_franka_reach.py --num-envs 64 --backend cpu
"""
import os
import sys

# Make `vision_rl` importable when run as a script (adds the repo root to path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_rl.train import main

if __name__ == "__main__":
    main()
