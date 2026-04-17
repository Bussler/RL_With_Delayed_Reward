"""Train PPO with RUDDER reward redistribution on the drone environment.

This script wraps the drone environment with :class:`RudderRewardWrapper`
so that the RUDDER return-predictor LSTM learns to attribute delayed rewards
(target destruction bonuses, completion bonuses) back to the earlier actions
that caused them.

Usage examples:
    # Train from scratch with default settings
    python scripts/train_skrl_ppo_rudder.py

    # Custom RUDDER parameters
    python scripts/train_skrl_ppo_rudder.py --rudder-alpha 0.5 --rudder-warmup 30

    # Load a pre-trained model for evaluation
    python scripts/train_skrl_ppo_rudder.py --model-path skrl/drone_ppo_tensorboard/models/rudder_lstm
"""

import argparse
import logging
import os

import torch
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO + RUDDER on drone environment")

    # --- Existing PPO args ------------------------------------------------
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to load pre-trained model. If empty, train from scratch.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="rudder_lstm",
        help="Name of the experiment for logging and model saving",
    )
    parser.add_argument(
        "--use-lstm",
        type=bool,
        default=True,
        help="Whether to use LSTM networks for PPO policy/value",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/drone_env/default_config.yaml",
        help="Path to the drone environment configuration YAML file",
    )
    parser.add_argument(
        "--training-length",
        type=int,
        default=300000,
        help="Maximum number of timesteps for training",
    )
    parser.add_argument(
        "--eval-render-interval",
        type=int,
        default=5,
        help="Interval for rendering during evaluation",
    )

    # --- RUDDER-specific args ---------------------------------------------
    parser.add_argument(
        "--rudder-alpha",
        type=float,
        default=0.3,
        help="Blending weight: alpha * r_rudder + (1-alpha) * r_original",
    )
    parser.add_argument(
        "--rudder-warmup",
        type=int,
        default=50,
        help="Number of episodes to collect before activating RUDDER redistribution",
    )
    parser.add_argument(
        "--rudder-train-interval",
        type=int,
        default=10,
        help="Train return predictor every N completed episodes",
    )
    parser.add_argument(
        "--rudder-train-epochs",
        type=int,
        default=5,
        help="Number of training passes per predictor training trigger",
    )
    parser.add_argument(
        "--rudder-batch-size",
        type=int,
        default=16,
        help="Episode batch size sampled from lessons buffer per epoch",
    )
    parser.add_argument(
        "--rudder-buffer-size",
        type=int,
        default=500,
        help="Maximum number of episodes stored in the lessons buffer",
    )
    parser.add_argument(
        "--rudder-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the RUDDER return predictor LSTM",
    )
    parser.add_argument(
        "--rudder-hidden",
        type=int,
        default=128,
        help="LSTM hidden size for the return predictor",
    )

    return parser.parse_args()


args = parse_args()

MODEL_PATH = args.model_path
TRAINING_LENGTH = args.training_length
EVAL_LENGTH = 1000
EVAL_RENDER_INTERVAL = args.eval_render_interval
EXPERIMENT_NAME = args.experiment_name
CONFIG_PATH = args.config_path

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

# ------------------------------------------------------------------
# Environment setup: DroneGymEnv → RudderRewardWrapper → skrl wrap
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_env = DroneGymEnv(drone_env_config=CONFIG_PATH, renderer="matplotlib", render_mode="rgb_array")

obs_dim = calculate_flattened_obs_space_size(base_env.observation_space)
action_dim = int(base_env.action_space.shape[0])

rudder_env = RudderRewardWrapper(
    base_env,
    obs_dim=obs_dim,
    action_dim=action_dim,
    alpha=args.rudder_alpha,
    warmup_episodes=args.rudder_warmup,
    train_interval=args.rudder_train_interval,
    train_epochs=args.rudder_train_epochs,
    batch_size=args.rudder_batch_size,
    buffer_size=args.rudder_buffer_size,
    predictor_lr=args.rudder_lr,
    predictor_hidden=args.rudder_hidden,
    device=device,
)

env = wrap_env(rudder_env)

# ------------------------------------------------------------------
# PPO configuration (same as train_skrl_ppo.py)
# ------------------------------------------------------------------
print(f"Using device: {device}")
print(f"Configuration: {CONFIG_PATH}")
print(f"Experiment name: {EXPERIMENT_NAME}")
print(
    f"RUDDER alpha={args.rudder_alpha}, warmup={args.rudder_warmup}, "
    f"train_interval={args.rudder_train_interval}, buffer_size={args.rudder_buffer_size}"
)

models = {}
if args.use_lstm:
    models["policy"] = PolicyLSTM(env.observation_space, env.action_space, device)
    models["value"] = ValueLSTM(env.observation_space, env.action_space, device)
else:
    models["policy"] = PolicyNW(env.observation_space, env.action_space, device)
    models["value"] = ValueNW(env.observation_space, env.action_space, device)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 2048 // env.num_envs
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = None
cfg["rewards_shaper"] = None
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
cfg["experiment"]["write_interval"] = 300
cfg["experiment"]["checkpoint_interval"] = 10000
cfg["experiment"]["directory"] = os.path.join("skrl", "drone_ppo_tensorboard")
cfg["experiment"]["experiment_name"] = EXPERIMENT_NAME

memory = RandomMemory(memory_size=cfg["rollouts"], num_envs=env.num_envs, device=device)

if args.use_lstm:
    agent = PPO_RNN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
else:
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

# ------------------------------------------------------------------
# Training or evaluation
# ------------------------------------------------------------------
if MODEL_PATH:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    agent.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
else:
    print("Starting training with RUDDER reward redistribution...")
    print(f"Observation space size: {obs_dim}")
    print(f"Action space: {env.action_space}")

    cfg_trainer = {"timesteps": TRAINING_LENGTH, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.train()

    models_dir = os.path.join("skrl", "drone_ppo_tensorboard", "models")
    os.makedirs(models_dir, exist_ok=True)
    agent.save(os.path.join(models_dir, EXPERIMENT_NAME))
    print(f"Training completed! Models saved to {models_dir}")

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
print("Running evaluation...")
agent.set_running_mode("eval")
observation, info = env.reset()

total_reward = 0
done = False
step = 0

with torch.no_grad():
    while not done and step < EVAL_LENGTH:
        action = agent.act(observation, timestep=step, timesteps=1)[0]
        next_observation, reward, terminated, truncated, info = env.step(action)
        reward = reward.item() if hasattr(reward, "item") else reward

        if step % EVAL_RENDER_INTERVAL == 0:
            env.render()
            print(f"Step: {step}, Action: {action}, Reward: {reward:.4f}, Total: {total_reward:.4f}")

        total_reward += reward
        observation = next_observation
        done = terminated or truncated
        step += 1

print(f"\nEpisode completed with total reward: {total_reward:.4f} in {step} steps")
env.close()
