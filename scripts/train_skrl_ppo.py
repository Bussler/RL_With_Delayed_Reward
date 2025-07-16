import argparse
import os

import torch

# Import our environment
from drone_environment.gym import DroneGymEnv, calculate_flattened_obs_space_size
from drone_environment.networks.ppo_policy import PolicyNW
from drone_environment.networks.ppo_value import ValueLSTM, ValueNW

# Import skrl components
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Set seed for reproducibility
set_seed(42)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train or evaluate PPO agent on drone environment")

    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to load pre-trained model from. If empty, training will start from scratch.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="lstm_no_mask_no_urgency_withScheduler2",
        help="Name of the experiment for logging and model saving",
    )

    parser.add_argument(
        "--use-lstm",
        type=bool,
        default=True,
        help="Whether to use LSTM for the ppo value function or not",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/drone_env/default_config.yaml",
        help="Path to the drone environment configuration YAML file",
    )

    parser.add_argument(
        "--training-length", type=int, default=300000, help="Maximum number of steps for training episode"
    )

    parser.add_argument(
        "--eval-render-interval", type=int, default=5, help="Interval for rendering during evaluation"
    )

    return parser.parse_args()


# Parse command line arguments
args = parse_args()

MODEL_PATH = args.model_path
TRAINING_LENGTH = args.training_length
EVAL_LENGTH = 1000
EVAL_RENDER_INTERVAL = args.eval_render_interval
EXPERIMENT_NAME = args.experiment_name
CONFIG_PATH = args.config_path

# Validate config path exists
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

# Create the environment
orig_env = DroneGymEnv(drone_env_config=CONFIG_PATH, renderer="matplotlib", render_mode="rgb_array")

# Wrap the environment for skrl
env = wrap_env(orig_env)


# Configure PPO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Configuration: {CONFIG_PATH}")
print(f"Experiment name: {EXPERIMENT_NAME}")

# Models
models = {}
models["policy"] = PolicyNW(env.observation_space, env.action_space, device)

if args.use_lstm:
    models["value"] = ValueLSTM(env.observation_space, env.action_space, device)
else:
    models["value"] = ValueNW(env.observation_space, env.action_space, device)

# Configure and create PPO agent
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 2048 // env.num_envs  # memory_size / num_envs
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
cfg["experiment"]["write_interval"] = 300  # seconds
cfg["experiment"]["checkpoint_interval"] = 10000  # timesteps
cfg["experiment"]["directory"] = os.path.join("skrl", "drone_ppo_tensorboard")  # log dir
cfg["experiment"]["experiment_name"] = EXPERIMENT_NAME

# Memory
memory = RandomMemory(memory_size=cfg["rollouts"], num_envs=env.num_envs, device=device)

# Create agent
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

if MODEL_PATH:
    # Validate model path exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Load agent and run evaluation episode
    agent.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
else:
    # Configure and create trainer
    print("Starting training...")
    print(f"Observation space size: {calculate_flattened_obs_space_size(env.observation_space)}")
    print(f"Action space: {env.action_space}")
    print(f"Number of targets: {env.num_targets}")

    cfg_trainer = {"timesteps": TRAINING_LENGTH, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # Start training
    trainer.train()

    # Save the trained model
    models_dir = os.path.join("skrl", "drone_ppo_tensorboard", "models")
    os.makedirs(models_dir, exist_ok=True)
    agent.save(os.path.join(models_dir, EXPERIMENT_NAME))

    print(f"Training completed! Models saved to {models_dir}")

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
