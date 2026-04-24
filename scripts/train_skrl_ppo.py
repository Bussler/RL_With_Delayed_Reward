"""Train PPO agent on the drone environment.

Usage examples:
    # Train from scratch with defaults
    python scripts/train_skrl_ppo.py

    # Use a custom config file
    python scripts/train_skrl_ppo.py --config configs/training/ppo.yaml

    # Override individual values (dot-notation, repeat as needed)
    python scripts/train_skrl_ppo.py --set ppo.learning_rate=1e-3 --set experiment.experiment_name=test

    # Load a pre-trained model for evaluation only
    python scripts/train_skrl_ppo.py --set model_path=skrl/drone_ppo_tensorboard/models/new_lstm
"""

import argparse
import os
import sys

# Ensure scripts/ is on sys.path so `config` module is importable when
# running with `python scripts/train_skrl_ppo.py` from the project root.
sys.path.insert(0, os.path.dirname(__file__))

import torch
from drone_environment.config import TrainConfig, load_config
from drone_environment.gym import DroneGymEnv, calculate_flattened_obs_space_size
from drone_environment.networks.ppo_policy import PolicyLSTM, PolicyNW
from drone_environment.networks.ppo_value import ValueLSTM, ValueNW

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

set_seed(42)

parser = argparse.ArgumentParser(description="Train or evaluate PPO agent on the drone environment")
parser.add_argument(
    "--config",
    type=str,
    default="configs/training/ppo.yaml",
    help="Path to YAML config file (default: configs/training/ppo.yaml)",
)
parser.add_argument(
    "--set",
    metavar="KEY=VALUE",
    action="append",
    default=[],
    dest="overrides",
    help="Override a config value using dot-notation, e.g. --set ppo.learning_rate=1e-3",
)
args = parser.parse_args()


# Config
cfg: TrainConfig = load_config(args.config, args.overrides, TrainConfig)

print(f"Configuration: {args.config}")
print(f"Experiment name: {cfg.experiment.experiment_name}")
print(f"Drone env config: {cfg.env.drone_config_path}")
if args.overrides:
    print(f"Overrides applied: {args.overrides}")

# Environment
orig_env = DroneGymEnv(
    drone_env_config=cfg.env.drone_config_path,
    renderer=cfg.env.renderer,
    render_mode=cfg.env.render_mode,
)
env = wrap_env(orig_env)


# Agent setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

models: dict[str, object] = {}
if cfg.ppo.use_lstm:
    models["policy"] = PolicyLSTM(env.observation_space, env.action_space, device)
    models["value"] = ValueLSTM(env.observation_space, env.action_space, device)
else:
    models["policy"] = PolicyNW(env.observation_space, env.action_space, device)
    models["value"] = ValueNW(env.observation_space, env.action_space, device)

ppo_cfg = PPO_DEFAULT_CONFIG.copy()
ppo_cfg["rollouts"] = cfg.ppo.rollouts // env.num_envs
ppo_cfg["learning_epochs"] = cfg.ppo.learning_epochs
ppo_cfg["mini_batches"] = cfg.ppo.mini_batches
ppo_cfg["discount_factor"] = cfg.ppo.discount_factor
ppo_cfg["lambda"] = cfg.ppo.lambda_
ppo_cfg["learning_rate"] = cfg.ppo.learning_rate
ppo_cfg["learning_rate_scheduler"] = KLAdaptiveRL
ppo_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": cfg.ppo.kl_threshold_scheduler}
ppo_cfg["grad_norm_clip"] = cfg.ppo.grad_norm_clip
ppo_cfg["ratio_clip"] = cfg.ppo.ratio_clip
ppo_cfg["value_clip"] = cfg.ppo.value_clip
ppo_cfg["clip_predicted_values"] = cfg.ppo.clip_predicted_values
ppo_cfg["entropy_loss_scale"] = cfg.ppo.entropy_loss_scale
ppo_cfg["value_loss_scale"] = cfg.ppo.value_loss_scale
ppo_cfg["kl_threshold"] = None
ppo_cfg["rewards_shaper"] = None
ppo_cfg["state_preprocessor"] = RunningStandardScaler
ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
ppo_cfg["value_preprocessor"] = RunningStandardScaler
ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
ppo_cfg["experiment"]["write_interval"] = cfg.experiment.write_interval
ppo_cfg["experiment"]["checkpoint_interval"] = cfg.experiment.checkpoint_interval
ppo_cfg["experiment"]["directory"] = cfg.experiment.directory
ppo_cfg["experiment"]["experiment_name"] = cfg.experiment.experiment_name

memory = RandomMemory(memory_size=ppo_cfg["rollouts"], num_envs=env.num_envs, device=device)

if cfg.ppo.use_lstm:
    agent = PPO_RNN(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
else:
    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

# Training or evaluation
if cfg.model_path:
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Model file not found: {cfg.model_path}")
    agent.load(cfg.model_path)
    print(f"Model loaded from {cfg.model_path}")
else:
    print("Starting training...")
    print(f"Observation space size: {calculate_flattened_obs_space_size(env.observation_space)}")
    print(f"Action space: {env.action_space}")
    print(f"Number of targets: {env.num_targets}")

    cfg_trainer = {"timesteps": cfg.training_length, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.train()

    models_dir = os.path.join(cfg.experiment.directory, "models")
    os.makedirs(models_dir, exist_ok=True)
    agent.save(os.path.join(models_dir, cfg.experiment.experiment_name))
    print(f"Training completed! Models saved to {models_dir}")

# Evaluation
print("Running evaluation...")
agent.set_running_mode("eval")
observation, info = env.reset()

total_reward = 0.0
done = False
step = 0

with torch.no_grad():
    rnn_states = None
    while not done and step < cfg.eval_length:
        action, _, outputs = agent.policy.act(
            {"states": observation.unsqueeze(0), "rnn": rnn_states}, role="policy"
        )

        rnn_states = outputs.get("rnn", None)

        next_observation, reward, terminated, truncated, info = env.step(action)
        reward = reward.item() if hasattr(reward, "item") else reward

        if step % cfg.eval_render_interval == 0:
            env.render()
            print(f"Step: {step}, Action: {action}, Reward: {reward:.4f}, Total: {total_reward:.4f}")

        total_reward += reward
        observation = next_observation
        done = terminated or truncated
        step += 1

print(f"\nEpisode completed with total reward: {total_reward:.4f} in {step} steps")
env.close()
