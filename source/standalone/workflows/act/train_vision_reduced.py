# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train ACT without privileged object / target position inputs."""

from __future__ import annotations

import sys

try:
    from .train import main as train_main
except ImportError:
    from train import main as train_main


def _inject_default_arg(flag: str, *values: str) -> None:
    if flag not in sys.argv[1:]:
        sys.argv.extend([flag, *values])


if __name__ == "__main__":
    _inject_default_arg("--exclude_low_dim_keys", "object_position", "target_object_position")
    _inject_default_arg("--aux_object_key", "object_position")
    _inject_default_arg("--aux_object_loss_weight", "5.0")
    train_main()
