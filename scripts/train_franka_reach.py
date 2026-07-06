#!/usr/bin/env python
"""Entry point: train vision PPO on Franka reach.

Examples:
    python scripts/train_franka_reach.py                # full run
    python scripts/train_franka_reach.py --small        # tiny (smoke) run
    python scripts/train_franka_reach.py --num-envs 64 --backend cpu
"""
from vision_rl.train import main

if __name__ == "__main__":
    main()
