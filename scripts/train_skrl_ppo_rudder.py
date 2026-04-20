"""Train PPO with RUDDER reward redistribution on the drone environment.

Usage examples:
    # Train from scratch with defaults
    python scripts/train_skrl_ppo_rudder.py

    # Use a custom config file
    python scripts/train_skrl_ppo_rudder.py --config configs/training/ppo_rudder.yaml

    # Override individual values (dot-notation, repeat as needed)
    python scripts/train_skrl_ppo_rudder.py --set rudder.alpha=0.5 --set rudder.warmup_episodes=30

    # Load a pre-trained model for evaluation only
    python scripts/train_skrl_ppo_rudder.py --set model_path=skrl/drone_ppo_tensorboard/models/rudder_lstm
"""

import argparse
import logging
import os
import sys

# Ensure scripts/ is on sys.path so `config` module is importable when
# running with `python scripts/train_skrl_ppo_rudder.py` from the project root.
sys.path.insert(0, os.path.dirname(__file__))

import torch
from config import TrainRudderConfig, load_config
from drone_environment.gym import DroneGymEnv, calculate_flattened_obs_space_size
from drone_environment.networks.ppo_policy import PolicyLSTM, PolicyNW
from drone_environment.networks.ppo_value import ValueLSTM, ValueNW
from drone_environment.rudder import RudderRewardWrapper

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

set_seed(42)

parser = argparse.ArgumentParser(description="Train PPO + RUDDER agent on the drone environment")
parser.add_argument(
    "--config",
    type=str,
    default="configs/training/ppo_rudder.yaml",
    help="Path to YAML config file (default: configs/training/ppo_rudder.yaml)",
)
parser.add_argument(
    "--set",
    metavar="KEY=VALUE",
    action="append",
    default=[],
    dest="overrides",
    help="Override a config value using dot-notation, e.g. --set rudder.alpha=0.5",
)
args = parser.parse_args()

# Config
cfg: TrainRudderConfig = load_config(args.config, args.overrides, TrainRudderConfig)

print(f"Configuration: {args.config}")
print(f"Experiment name: {cfg.experiment.experiment_name}")
print(f"Drone env config: {cfg.env.drone_config_path}")
print(
    f"RUDDER alpha={cfg.rudder.alpha}, warmup={cfg.rudder.warmup_episodes}, "
    f"train_interval={cfg.rudder.train_interval}, buffer_size={cfg.rudder.buffer_size}"
)
if args.overrides:
    print(f"Overrides applied: {args.overrides}")

# Environment: DroneGymEnv → RudderRewardWrapper -> skrl wrap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_env = DroneGymEnv(
    drone_env_config=cfg.env.drone_config_path,
    renderer=cfg.env.renderer,
    render_mode=cfg.env.render_mode,
)

obs_dim = calculate_flattened_obs_space_size(base_env.observation_space)
action_dim = int(base_env.action_space.shape[0])

rudder_env = RudderRewardWrapper(
    base_env,
    obs_dim=obs_dim,
    action_dim=action_dim,
    alpha=cfg.rudder.alpha,
    warmup_episodes=cfg.rudder.warmup_episodes,
    train_interval=cfg.rudder.train_interval,
    train_epochs=cfg.rudder.train_epochs,
    batch_size=cfg.rudder.batch_size,
    buffer_size=cfg.rudder.buffer_size,
    predictor_lr=cfg.rudder.predictor_lr,
    predictor_hidden=cfg.rudder.predictor_hidden,
    device=device,
)

env = wrap_env(rudder_env)

# Agent setup
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

# Training
if cfg.model_path:
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Model file not found: {cfg.model_path}")
    agent.load(cfg.model_path)
    print(f"Model loaded from {cfg.model_path}")
else:
    print("Starting training with RUDDER reward redistribution...")
    print(f"Observation space size: {obs_dim}")
    print(f"Action space: {env.action_space}")

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
    while not done and step < cfg.eval_length:
        action = agent.act(observation, timestep=step, timesteps=1)[0]
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
