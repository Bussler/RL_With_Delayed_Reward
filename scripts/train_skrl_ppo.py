import os

import gymnasium as gym
import torch

# Import our environment
from drone_environment.gym import DroneGymEnv, calculate_flattened_obs_space_size
from torch import nn

# Import skrl components
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Set seed for reproducibility
set_seed(42)

MODEL_PATH = ""  # "skrl/drone_ppo_tensorboard/models/agent"
EVAL_LENGTH = 1000
EVAL_RENDER_INTERVAL = 5

# Create the environment
orig_env = DroneGymEnv(renderer="matplotlib", render_mode="rgb_array")

# Wrap the environment for skrl
env = wrap_env(orig_env)


# Define models for PPO
class Actor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box,
        device: torch.device,
        *,
        clip_actions: bool = False,
    ) -> None:
        """Initialize the Actor model.

        Args:
            observation_space: The observation space of the environment
            action_space: The action space of the environment
            device: The device to run the model on
            clip_actions: Whether to clip actions to their bounds
        """
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions)

        input_size = int(calculate_flattened_obs_space_size(observation_space))

        # Define network architecture
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs: dict, role: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute the policy distribution parameters.

        Args:
            inputs: Dictionary containing input states. States are already flattened.
            role: Role of the model

        Returns:
            Tuple containing mean, log_std_parameter and features dictionary
        """
        x = inputs["states"]

        # Forward pass through network
        features = self.net(x)

        # Check for NaN values in features tensor
        if torch.isnan(features).any():
            print("WARNING: NaN values detected in features tensor!")

        # Compute mean
        mean = self.mean_layer(features)

        # Apply tanh to constrain actions to [-1, 1] range
        mean = torch.tanh(mean)  # TODO test if this is a problem!

        return mean, self.log_std_parameter, {"features": features}


class ValueNW(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        clip_actions: bool = False,
    ) -> None:
        """Initialize the Critic model.

        Args:
            observation_space: The observation space
            action_space: The action space
            device: The device to run the model on
            clip_actions: Whether to clip actions
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        input_size = int(calculate_flattened_obs_space_size(observation_space))

        # Define network architecture
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs: dict, role: str) -> tuple[torch.Tensor, dict]:
        """Compute the value function for the given inputs.

        Args:
            inputs: Dictionary containing states. States are already flattened.
            role: Role of the model

        Returns:
            Tuple containing the computed value and an empty dict
        """
        x = inputs["states"]

        # Forward pass through network
        return self.net(x), {}


# Configure PPO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Models
models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device)
models["value"] = ValueNW(env.observation_space, env.action_space, device)

# Configure and create PPO agent
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 2048 // env.num_envs  # memory_size / num_envs
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = None
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
cfg["experiment"]["experiment_name"] = "DroneGym_PPO"  # Experiment name

# Memory
memory = RandomMemory(memory_size=cfg["rollouts"], num_envs=env.num_envs, device=device)

# Create agent
agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

if MODEL_PATH:
    # Load agent and run evaluation episode
    agent.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
else:
    # Configure and create trainer
    print("Starting training...")
    print(f"Observation space size: {calculate_flattened_obs_space_size(env.observation_space)}")
    print(f"Action space: {env.action_space}")
    print(f"Number of targets: {env.num_targets}")

    cfg_trainer = {"timesteps": 300000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # Start training
    trainer.train()

    # Save the trained model
    models_dir = os.path.join("skrl", "drone_ppo_tensorboard", "models")
    os.makedirs(models_dir, exist_ok=True)
    agent.save(os.path.join(models_dir, "agent_no_high_masking_no_urgency"))

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
