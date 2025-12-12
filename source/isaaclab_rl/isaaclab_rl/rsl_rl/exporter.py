# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
from typing import Tuple


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        self.is_imitation = policy.is_imitation
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
            if self.is_imitation:
                self.enc = copy.deepcopy(policy.enc_a)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        elif hasattr(policy, "head_a"):
            self.actor = copy.deepcopy(policy.head_a)
            if self.is_imitation:
                self.enc = copy.deepcopy(policy.enc_a)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            if self.rnn_type == "lstm":
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # set up tcn-based actor network
        if self.is_imitation:
            self.enc.cpu()
            self.forward = self.forward_tcn
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward_gru(self, x):
        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self.actor(x)

    def forward_tcn(self, x: Tuple[torch.Tensor, torch.Tensor]):
        state_history, state = x
        state_history = self.normalizer(state_history)
        state = self.normalizer(state)
        state_history = state_history.permute(0, 2, 1).contiguous()  # Change to (batch, channels, time) for TCN
        x = self.enc(state_history).mean(dim=-1)  # Global average pooling over time dimension
        x = torch.cat([x, state], dim=-1)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        self.is_imitation = policy.is_imitation
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
            if self.is_imitation:
                self.enc = copy.deepcopy(policy.enc_a)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        elif hasattr(policy, "head_a"):
            self.actor = copy.deepcopy(policy.head_a)
            if self.is_imitation:
                self.enc = copy.deepcopy(policy.enc_a)
                self.state_history_input_shape = policy.state_history_shape
                self.state_size = policy.num_actor_state_obs
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # set up tcn-based actor network
        if self.is_imitation:
            self.enc.cpu()
            self.forward = self.forward_tcn
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward_tcn(self, x_history, x):
        x_history = self.normalizer(x_history)
        x = self.normalizer(x)
        x_history = x_history.permute(0, 2, 1).contiguous()  # Change to (batch, channels, time) for TCN
        encoded_x = self.enc(x_history).mean(dim=-1)  # Global average pooling over time dimension
        x = torch.cat([encoded_x, x], dim=-1)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        elif self.is_imitation:
            print("state_history_input_shape:", self.state_history_input_shape)
            history_shape = tuple(self.state_history_input_shape)
            obs_history = torch.zeros(1, *history_shape)
            obs = torch.zeros(1, self.state_size)
            torch.onnx.export(
                self,
                (obs_history, obs),
                os.path.join(path, filename),
                export_params=True,
                opset_version=opset_version,
                verbose=self.verbose,
                input_names=["state_history", "state"],
                output_names=["actions"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=opset_version,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
