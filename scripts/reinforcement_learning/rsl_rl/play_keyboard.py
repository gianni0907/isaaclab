# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint with RSL-RL while steering commands from keyboard."""


"""Launch Isaac Sim Simulator first."""

# Teleoperation scaling parameters (keyboard-only control)
SPEED_SCALE_DEFAULT = 1.0
SPEED_SCALE_STEP = 0.1
SPEED_SCALE_MIN = 0.0
SPEED_SCALE_MAX = 3.0
SPEED_INCREASE_KEY = "W"
SPEED_DECREASE_KEY = "Q"

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RSL-RL checkpoint with keyboard-driven commands.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during the session.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
	"--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
	"--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
	"--use_pretrained_checkpoint",
	action="store_true",
	help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
	"--command_term",
	type=str,
	default="base_velocity",
	help="Command term to override with keyboard input (default: base_velocity).",
)
parser.add_argument("--toggle_key", type=str, default="SPACE", help="Key used to toggle teleoperation on/off.")
parser.add_argument("--reset_key", type=str, default="R", help="Key used to reset the environment.")
parser.add_argument("--vx_sensitivity", type=float, default=0.8, help="Keyboard sensitivity for forward motion.")
parser.add_argument("--vy_sensitivity", type=float, default=0.4, help="Keyboard sensitivity for lateral motion.")
parser.add_argument("--omega_sensitivity", type=float, default=1.0, help="Keyboard sensitivity for yaw motion.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
	args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
	DirectMARLEnv,
	DirectMARLEnvCfg,
	DirectRLEnvCfg,
	ManagerBasedRLEnvCfg,
	multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
	"""Play with an RSL-RL agent while steering the command term from the keyboard."""
	# grab task name for checkpoint path
	task_name = args_cli.task.split(":")[-1]
	train_task_name = task_name.replace("-Play", "")

	# override configurations with non-hydra CLI arguments
	agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
	env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

	# set the environment seed
	env_cfg.seed = agent_cfg.seed
	env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

	# specify directory for logging experiments
	log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
	log_root_path = os.path.abspath(log_root_path)
	print(f"[INFO] Loading experiment from directory: {log_root_path}")
	if args_cli.use_pretrained_checkpoint:
		resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
		if not resume_path:
			print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
			return
	elif args_cli.checkpoint:
		resume_path = retrieve_file_path(args_cli.checkpoint)
	else:
		resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

	log_dir = os.path.dirname(resume_path)

	# set the log directory for the environment (works for all environment types)
	env_cfg.log_dir = log_dir
	env_cfg.viewer.origin_type = "asset_root"   # look relative to the robot instance
	env_cfg.viewer.asset_name = "robot"         # matches the asset registered in the scene
	env_cfg.viewer.env_index = 0                # which parallel env to focus on
	env_cfg.viewer.eye = (3.0, 3.0, 1.2)        # camera position in meters
	env_cfg.viewer.lookat = (0.0, 0.0, 0.5)     # focal point (robot torso)

	# keep the session running indefinitely (skip time-out truncations)
	if hasattr(env_cfg, "terminations") and getattr(env_cfg.terminations, "time_out", None) is not None:
		env_cfg.terminations.time_out = None

	# create isaac environment
	env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

	# convert to single-agent instance if required by the RL algorithm
	if isinstance(env.unwrapped, DirectMARLEnv):
		env = multi_agent_to_single_agent(env)

	# wrap for video recording
	if args_cli.video:
		video_kwargs = {
			"video_folder": os.path.join(log_dir, "videos", "play"),
			"step_trigger": lambda step: step == 0,
			"video_length": args_cli.video_length,
			"disable_logger": True,
		}
		print("[INFO] Recording videos during keyboard play.")
		print_dict(video_kwargs, nesting=4)
		env = gym.wrappers.RecordVideo(env, **video_kwargs)

	# wrap around environment for rsl-rl
	env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

	print(f"[INFO]: Loading model checkpoint from: {resume_path}")
	# load previously trained model
	if agent_cfg.class_name == "OnPolicyRunner":
		runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
	elif agent_cfg.class_name == "DistillationRunner":
		runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
	else:
		raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
	runner.load(resume_path)

	# obtain the trained policy for inference
	policy = runner.get_inference_policy(device=env.unwrapped.device)

	# extract the neural network module
	try:
		policy_nn = runner.alg.policy
	except AttributeError:
		policy_nn = runner.alg.actor_critic

	# extract the normalizer
	if hasattr(policy_nn, "actor_obs_normalizer"):
		normalizer = policy_nn.actor_obs_normalizer
	elif hasattr(policy_nn, "student_obs_normalizer"):
		normalizer = policy_nn.student_obs_normalizer
	else:
		normalizer = None

	# export policy to onnx/jit
	export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
	export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
	export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

	dt = env.unwrapped.step_dt

	# prepare keyboard teleoperation
	teleop_cfg = Se2KeyboardCfg(
		v_x_sensitivity=args_cli.vx_sensitivity,
		v_y_sensitivity=args_cli.vy_sensitivity,
		omega_z_sensitivity=args_cli.omega_sensitivity,
		sim_device=env.unwrapped.device,
	)
	teleop = Se2Keyboard(teleop_cfg)
	teleop.reset()

	try:
		command_term = env.unwrapped.command_manager.get_term(args_cli.command_term)
	except (KeyError, AttributeError) as exc:
		print(
			"[ERROR] The environment does not expose the requested command term for manual control."
			f" Requested term: '{args_cli.command_term}'."
		)
		print(f"        {exc}")
		env.close()
		simulation_app.close()
		return

	command_tensor = torch.zeros_like(command_term.command)

	teleoperation_active = True
	should_reset_env = False

	toggle_key = args_cli.toggle_key.upper()
	reset_key = args_cli.reset_key.upper()
	speed_increase_key = SPEED_INCREASE_KEY.upper()
	speed_decrease_key = SPEED_DECREASE_KEY.upper()
	command_scale = SPEED_SCALE_DEFAULT

	# Disable automatic resampling from the command manager so manual inputs persist.
	if hasattr(env.unwrapped, "command_manager") and hasattr(env.unwrapped.command_manager, "compute"):
		env.unwrapped.command_manager.compute = lambda dt: None

	def _clamp_scale(value: float) -> float:
		return max(SPEED_SCALE_MIN, min(SPEED_SCALE_MAX, value))

	def toggle_teleoperation():
		nonlocal teleoperation_active
		teleoperation_active = not teleoperation_active
		state = "enabled" if teleoperation_active else "paused"
		print(f"[INFO] Teleoperation {state} (press {toggle_key} to toggle).")

	def request_reset():
		nonlocal should_reset_env
		should_reset_env = True
		print("[INFO] Environment reset requested.")

	def increase_command_scale():
		nonlocal command_scale
		command_scale = _clamp_scale(command_scale + SPEED_SCALE_STEP)
		print(
			f"[INFO] Velocity multiplier increased to {command_scale:.2f} (max {SPEED_SCALE_MAX:.2f})."
		)

	def decrease_command_scale():
		nonlocal command_scale
		command_scale = _clamp_scale(command_scale - SPEED_SCALE_STEP)
		print(
			f"[INFO] Velocity multiplier decreased to {command_scale:.2f} (min {SPEED_SCALE_MIN:.2f})."
		)

	try:
		teleop.add_callback(toggle_key, toggle_teleoperation)
	except (ValueError, TypeError):
		print(f"[WARN] Failed to bind toggle key '{toggle_key}'.")
	try:
		teleop.add_callback(reset_key, request_reset)
	except (ValueError, TypeError):
		print(f"[WARN] Failed to bind reset key '{reset_key}'.")
	try:
		teleop.add_callback(speed_increase_key, increase_command_scale)
	except (ValueError, TypeError):
		print(f"[WARN] Failed to bind speed increase key '{speed_increase_key}'.")
	try:
		teleop.add_callback(speed_decrease_key, decrease_command_scale)
	except (ValueError, TypeError):
		print(f"[WARN] Failed to bind speed decrease key '{speed_decrease_key}'.")

	print("[INFO] Keyboard control ready.")
	print("       Forward/back: Arrow Up / Arrow Down")
	print("       Strafe: Arrow Left / Arrow Right")
	print("       Yaw: Z / X")
	print(f"       Toggle teleoperation: {toggle_key}")
	print(f"       Reset environments: {reset_key}")
	print(
		f"       Adjust velocity multiplier (current {command_scale:.2f}): +{speed_increase_key} / -{speed_decrease_key}"
	)

	# helper to push current command to the command term
	def apply_command_to_term(command: torch.Tensor):
		command_term.command.copy_(command)
		command_term.time_left.fill_(float("inf"))

	apply_command_to_term(command_tensor)

	# simulate environment
	while simulation_app.is_running():
		start_time = time.time()
		with torch.inference_mode():
			if teleoperation_active:
				input_command = teleop.advance()
				if torch.allclose(input_command, torch.zeros_like(input_command), atol=1.0e-6):
					command_tensor.zero_()
				else:
					scaled_command = input_command * command_scale
					command_tensor[:] = scaled_command.view(1, -1).repeat(env.num_envs, 1)
			else:
				command_tensor.zero_()

			apply_command_to_term(command_tensor)

			observations = env.get_observations()
			actions = policy(observations)
			_, _, dones, _ = env.step(actions)
			policy_nn.reset(dones)

			if should_reset_env:
				env.reset()
				policy_nn.reset(torch.ones_like(dones))
				command_tensor.zero_()
				teleop.reset()
				apply_command_to_term(command_tensor)
				should_reset_env = False

		# time delay for real-time evaluation
		sleep_time = dt - (time.time() - start_time)
		if args_cli.real_time and sleep_time > 0:
			time.sleep(sleep_time)

	# close the simulator
	env.close()


if __name__ == "__main__":
	# run the main function
	main()
	# close sim app
	simulation_app.close()
