# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum utilities specific to the Spot locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class ActuatorShutdownCurriculumCfg(CurriculumTermCfg):
    """Configuration container for the actuator shutdown curriculum."""

    event_name: str = "actuator_shutdown"
    command_name: str = "base_velocity"
    initial_scale: float = 1.0
    min_scale: float = 0.2
    max_scale: float = 1.0
    scale_step: float = 0.05
    recovery_step: float = 0.01
    tracking_error_threshold: float = 0.3
    success_ratio: float = 0.7
    update_every: int = 1024
    min_command_speed: float = 0.2
    weakening_joint_indices: Sequence[int] | None = None


class ActuatorShutdownCurriculum(ManagerTermBase):
    """Curriculum that scales per-joint actuator shutdown bounds based on tracking performance.

    Each environment samples one joint from ``weakening_joint_indices`` that will experience reduced gains while all
    other joints retain their nominal actuator values (scale ``1``). When the agent tracks commands reliably, the stored
    minimum scale for the faulted joint is reduced by ``scale_step`` down to ``min_scale``. If tracking quality drops,
    the same joint recovers towards its nominal range in increments of ``recovery_step``. Joint-specific minima persist
    across episodes so future faults reuse the previously achieved bounds.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params

        self._event_name: str = getattr(cfg, "event_name", params.get("event_name", "actuator_shutdown"))
        self._asset_cfg: SceneEntityCfg = params.get("asset_cfg", SceneEntityCfg("robot"))
        self._command_name: str = params.get("command_name", getattr(cfg, "command_name", "base_velocity"))

        self._initial_scale: float = float(getattr(cfg, "initial_scale", params.get("initial_scale", 1.0)))
        self._min_scale: float = float(getattr(cfg, "min_scale", params.get("min_scale", 0.2)))
        self._max_scale: float = float(getattr(cfg, "max_scale", params.get("max_scale", 1.0)))
        self._scale_step: float = float(getattr(cfg, "scale_step", params.get("scale_step", 0.05)))
        self._recovery_step: float = float(getattr(cfg, "recovery_step", params.get("recovery_step", 0.01)))
        self._tracking_error_threshold: float = float(
            getattr(cfg, "tracking_error_threshold", params.get("tracking_error_threshold", 0.3))
        )
        self._success_ratio: float = float(getattr(cfg, "success_ratio", params.get("success_ratio", 0.7)))
        self._update_every: int = int(getattr(cfg, "update_every", params.get("update_every", 1024)))
        self._min_command_speed: float = float(
            getattr(cfg, "min_command_speed", params.get("min_command_speed", 0.2))
        )

        weakening_joint_indices: Sequence[int] | None = getattr(
            cfg, "weakening_joint_indices", params.get("weakening_joint_indices")
        )

        if self._min_scale > self._max_scale:
            raise ValueError("Actuator shutdown curriculum requires min_scale <= max_scale.")
        if not (self._min_scale <= self._initial_scale <= self._max_scale):
            raise ValueError("initial_scale must lie within [min_scale, max_scale].")

        self._event_cfg = env.event_manager.get_term_cfg(self._event_name)
        if self._event_cfg is None:
            raise ValueError(f"Event term '{self._event_name}' not found. Unable to configure curriculum.")

        stiff_params = self._event_cfg.params.get("stiffness_distribution_params")
        damp_params = self._event_cfg.params.get("damping_distribution_params")
        if stiff_params is None or damp_params is None:
            raise ValueError(
                "Actuator shutdown curriculum expects both stiffness and damping distribution parameters to be present."
            )

        device = env.device
        self._device = device
        self._num_envs = env.scene.num_envs

        self._stiffness_base_lower = torch.tensor(stiff_params[0], dtype=torch.float32, device=device)
        self._stiffness_base_upper = torch.tensor(stiff_params[1], dtype=torch.float32, device=device)
        self._damping_base_lower = torch.tensor(damp_params[0], dtype=torch.float32, device=device)
        self._damping_base_upper = torch.tensor(damp_params[1], dtype=torch.float32, device=device)

        if not torch.allclose(self._stiffness_base_lower, self._damping_base_lower):
            raise ValueError("Actuator shutdown curriculum expects identical stiffness and damping lower bounds.")
        if not torch.allclose(self._stiffness_base_upper, self._damping_base_upper):
            raise ValueError("Actuator shutdown curriculum expects identical stiffness and damping upper bounds.")

        num_joints = self._stiffness_base_lower.numel()
        if weakening_joint_indices is None:
            weakening_mask = torch.ones(num_joints, dtype=torch.bool, device=device)
        else:
            indices_tensor = torch.as_tensor(weakening_joint_indices, dtype=torch.long, device=device)
            if indices_tensor.numel() == 0:
                raise ValueError("weakening_joint_indices cannot be empty when provided.")
            if (indices_tensor < 0).any() or (indices_tensor >= num_joints).any():
                raise ValueError("weakening_joint_indices contains out-of-range joint indices.")
            weakening_mask = torch.zeros(num_joints, dtype=torch.bool, device=device)
            weakening_mask[indices_tensor.unique(sorted=True)] = True

        self._weakening_mask = weakening_mask
        self._weakening_indices = torch.nonzero(self._weakening_mask, as_tuple=False).flatten()

        self._fault_lower = torch.ones(num_joints, dtype=torch.float32, device=device)
        self._fault_upper = torch.ones(num_joints, dtype=torch.float32, device=device)
        if self._weakening_indices.numel():
            initial_vals = torch.full((self._weakening_indices.numel(),), self._initial_scale, device=device)
            initial_vals = torch.clamp(initial_vals, self._min_scale, self._max_scale)
            self._fault_lower[self._weakening_mask] = initial_vals

            upper_vals = torch.full_like(initial_vals, self._max_scale)
            upper_vals = torch.clamp(upper_vals, self._min_scale, self._max_scale)
            self._fault_upper[self._weakening_mask] = upper_vals

        self._active_joints = torch.full((self._num_envs,), -1, dtype=torch.long, device=device)
        self._joint_fault_mask = torch.zeros((self._num_envs, num_joints), dtype=torch.bool, device=device)

        self._last_update_step: int = -self._update_every

        self._assign_new_joints(torch.arange(self._num_envs, device=device))
        self._apply_scale()

    def reset(self, env_ids: Sequence[int] | None = None):  # noqa: D401 - inherited docstring is sufficient
        if env_ids is None:
            env_ids_tensor = torch.arange(self._num_envs, device=self._device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

        if env_ids_tensor.numel() == 0:
            return

        self._last_update_step = -self._update_every
        self._assign_new_joints(env_ids_tensor)
        self._apply_scale()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        asset_cfg: SceneEntityCfg | None = None,
        command_name: str | None = None,
    ) -> dict[str, float]:
        del env_ids  # decisions rely on aggregate statistics

        if self._weakening_indices.numel() == 0:
            return {"mean_scale": 0.0}

        if self._update_every > 0:
            if env.common_step_counter - self._last_update_step < self._update_every:
                return self._current_metrics()
            self._last_update_step = env.common_step_counter

        asset_cfg = asset_cfg or self._asset_cfg
        command_name = command_name or self._command_name

        robot = env.scene[asset_cfg.name]
        commands = env.command_manager.get_command(command_name)

        lin_vel = robot.data.root_lin_vel_w[:, :2]
        cmd_vel = commands[:, :2]

        command_mag = torch.linalg.norm(cmd_vel, dim=1)
        valid_mask = command_mag > self._min_command_speed
        if valid_mask.any():
            velocity_error = torch.linalg.norm(lin_vel[valid_mask] - cmd_vel[valid_mask], dim=1)
        else:
            velocity_error = torch.linalg.norm(lin_vel - cmd_vel, dim=1)

        success_ratio = (velocity_error < self._tracking_error_threshold).float().mean().item()
        if success_ratio >= self._success_ratio:
            updated = self._increase_difficulty()
        else:
            updated = self._decrease_difficulty()

        if updated:
            self._apply_scale()

        return self._current_metrics()

    def _apply_scale(self):
        lower = torch.ones_like(self._stiffness_base_lower)
        upper = torch.ones_like(self._stiffness_base_upper)

        if self._weakening_indices.numel():
            clamped_lower = torch.clamp(self._fault_lower[self._weakening_mask], self._min_scale, self._max_scale)
            self._fault_lower[self._weakening_mask] = clamped_lower
            lower[self._weakening_mask] = clamped_lower

            clamped_upper = torch.clamp(self._fault_upper[self._weakening_mask], self._min_scale, self._max_scale)
            self._fault_upper[self._weakening_mask] = clamped_upper
            upper[self._weakening_mask] = clamped_upper

        params_tuple = (
            tuple(lower.cpu().tolist()),
            tuple(upper.cpu().tolist()),
        )
        self._event_cfg.params["stiffness_distribution_params"] = params_tuple
        self._event_cfg.params["damping_distribution_params"] = params_tuple
        self._event_cfg.params["joint_fault_mask"] = self._joint_fault_mask.detach().to(device="cpu").tolist()

        self._env.event_manager.set_term_cfg(self._event_name, self._event_cfg)

    def _increase_difficulty(self) -> bool:
        if self._weakening_indices.numel() == 0:
            return False

        active = self._active_joints[self._active_joints >= 0]
        if active.numel() == 0:
            return False

        updated = False
        for joint_idx in torch.unique(active).tolist():
            current_value = float(self._fault_lower[joint_idx].item())
            new_value = max(current_value - self._scale_step, self._min_scale)
            if new_value < current_value - 1e-6:
                self._fault_lower[joint_idx] = new_value
                updated = True
        return updated

    def _decrease_difficulty(self) -> bool:
        if self._weakening_indices.numel() == 0:
            return False

        active = self._active_joints[self._active_joints >= 0]
        if active.numel() == 0:
            return False

        updated = False
        for joint_idx in torch.unique(active).tolist():
            current_value = float(self._fault_lower[joint_idx].item())
            target_upper = float(self._fault_upper[joint_idx].item())
            new_value = min(current_value + self._recovery_step, target_upper)
            if new_value > current_value + 1e-6:
                self._fault_lower[joint_idx] = new_value
                updated = True
        return updated

    def _assign_new_joints(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        if self._weakening_indices.numel() == 0:
            self._active_joints[env_ids] = -1
            self._joint_fault_mask[env_ids] = False
            return

        for env_id in env_ids.tolist():
            joint_idx = int(
                self._weakening_indices[
                    torch.randint(0, self._weakening_indices.numel(), (1,), device=self._device)
                ].item()
            )
            self._active_joints[env_id] = joint_idx
            self._joint_fault_mask[env_id].zero_()
            self._joint_fault_mask[env_id, joint_idx] = True

    def _current_value_metric(self) -> float:
        if self._weakening_indices.numel() == 0:
            return 0.0

        values = self._fault_lower[self._weakening_mask]
        if values.numel() == 0:
            return float(self._initial_scale)

        return float(values.mean().item())

    def _current_metrics(self) -> dict[str, float]:
        metrics = {"mean_scale": self._current_value_metric()}

        if self._weakening_indices.numel() == 0:
            return metrics

        for joint_idx in self._weakening_indices.tolist():
            metrics[f"joint_{joint_idx}"] = float(self._fault_lower[joint_idx].item())

        return metrics
