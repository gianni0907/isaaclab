# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation utilities that are specific to the Spot locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def actuator_shutdown_fault_mask(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    event_name: str = "actuator_shutdown",
) -> torch.Tensor:
    """Return the per-environment actuator fault mask used by the shutdown randomization event.

    The mask is created by the curriculum to mark which joint (if any) experiences reduced gains in a given
    environment. When the curriculum has not set the mask yet, a zero tensor is returned.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    device = robot.device

    num_envs = env.scene.num_envs
    num_joints = robot.data.default_joint_stiffness.shape[1]
    empty_mask = torch.zeros((num_envs, num_joints), dtype=torch.float32, device=device)

    event_cfg = env.event_manager.get_term_cfg(event_name)
    if event_cfg is None:
        return empty_mask

    mask = event_cfg.params.get("joint_fault_mask")
    if mask is None:
        return empty_mask

    try:
        mask_tensor = torch.as_tensor(mask, device=device, dtype=torch.bool)
    except TypeError:
        return empty_mask

    if mask_tensor.ndim != 2:
        return empty_mask

    if mask_tensor.shape[0] != num_envs or mask_tensor.shape[1] != num_joints:
        return empty_mask

    return mask_tensor.float()
