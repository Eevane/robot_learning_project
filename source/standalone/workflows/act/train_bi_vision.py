# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train ACT on the Bi-ACT vision observation variant."""

from __future__ import annotations

import os
import sys

try:
    from .train import main as train_main
except ImportError:
    from train import main as train_main


def _inject_default_arg(flag: str, value: str) -> None:
    if flag not in sys.argv[1:]:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _inject_default_arg("--task", "Isaac-Lift-Needle-PSM-IK-Abs-Bi-Vision-v0")
    _inject_default_arg(
        "--dataset",
        os.path.join("logs", "act", "Isaac-Lift-Needle-PSM-IK-Abs-Bi-Vision-v0", "act_dataset_wider"),
    )
    train_main()
