import argparse
import os

import gymnasium as gym
import torch

# Import our environment
from drone_environment.gym import DroneGymEnv, calculate_flattened_obs_space_size
from torch import nn

# Import skrl components
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
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
        default="lstm_no_mask_no_urgency",
        help="Name of the experiment for logging and model saving",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/drone_env/default_config.yaml",
        help="Path to the drone environment configuration YAML file",
    )

    parser.add_argument(
        "--eval-length", type=int, default=1000, help="Maximum number of steps for evaluation episode"
    )

    parser.add_argument(
        "--eval-render-interval", type=int, default=5, help="Interval for rendering during evaluation"
    )

    return parser.parse_args()


# Parse command line arguments
args = parse_args()

MODEL_PATH = args.model_path
EVAL_LENGTH = args.eval_length
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


# Define models for PPO
class PolicyNW(GaussianMixin, Model):
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

        # Compute mean
        mean = self.mean_layer(features)

        # Apply tanh to constrain actions to [-1, 1] range
        mean = torch.tanh(mean)  # TODO test if this is a problem!

        return mean, self.log_std_parameter, {"features": features}


# class PolicyLSTMNW(GaussianMixin, Model):
#     def __init__(
#         self,
#         observation_space: gym.spaces.Dict,
#         action_space: gym.spaces.Box,
#         device: torch.device,
#         *,
#         clip_actions: bool = False,
#         lstm_hidden_size: int = 128,
#         lstm_num_layers: int = 1,
#         num_envs: int = 1,
#         sequence_length: int = 128,
#     ) -> None:
#         """Initialize the Actor model with LSTM."""
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions)

#         input_size = int(calculate_flattened_obs_space_size(observation_space))
#         self.lstm_hidden_size = lstm_hidden_size
#         self.lstm_num_layers = lstm_num_layers
#         self.sequence_length = sequence_length
#         self.num_envs = num_envs

#         # LSTM layer
#         self.lstm = nn.LSTM(
#             input_size=128,
#             hidden_size=lstm_hidden_size,
#             num_layers=lstm_num_layers,
#             batch_first=True,
#         )

#         # Pre-LSTM layers
#         self.pre_lstm = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#         )

#         # Output layers for mean and log_std
#         self.mean_layer = nn.Linear(lstm_hidden_size, self.num_actions)
#         self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

#     def get_specification(self) -> dict:
#         """Get model specification including RNN information."""
#         # spec = super().get_specification()
#         # spec.update(
#         #     {
#         #         "rnn": {
#         #             "sizes": [self.lstm_hidden_size] * self.lstm_num_layers,
#         #             "layers": self.lstm_num_layers,
#         #         }
#         #     }
#         # )
#         # return spec

#         return {
#             "rnn": {
#                 "sequence_length": self.sequence_length,
#                 "sizes": [
#                     (
#                         self.lstm_num_layers,
#                         self.num_envs,
#                         self.lstm_hidden_size,
#                     ),  # hidden states (D ∗ num_layers, N, Hout)
#                     (self.lstm_num_layers, self.num_envs, self.lstm_hidden_size),
#                 ],
#             }
#         }

#     def compute(self, inputs: dict, role: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
#         """Compute the policy distribution parameters."""
#         x = inputs["states"]
#         batch_size = x.shape[0]

#         # Handle RNN states more robustly
#         if "rnn" in inputs and inputs["rnn"] is not None and len(inputs["rnn"]) == 2:
#             try:
#                 h_state = inputs["rnn"][0].view(self.lstm_num_layers, batch_size, self.lstm_hidden_size)
#                 c_state = inputs["rnn"][1].view(self.lstm_num_layers, batch_size, self.lstm_hidden_size)
#                 hidden_states = (h_state.to(self.device), c_state.to(self.device))
#             except (RuntimeError, IndexError) as e:
#                 print(f"Warning: RNN state reshaping failed ({e}), using zero states")
#                 hidden_states = self._get_zero_hidden_states(batch_size)
#         else:
#             hidden_states = self._get_zero_hidden_states(batch_size)

#         # Pre-LSTM processing
#         x = self.pre_lstm(x)

#         # Add sequence dimension for LSTM
#         x = x.unsqueeze(1)

#         # LSTM forward pass
#         lstm_out, new_hidden_states = self.lstm(x, hidden_states)

#         # Remove sequence dimension
#         lstm_out = lstm_out.squeeze(1)

#         # Compute mean
#         mean = self.mean_layer(lstm_out)
#         mean = torch.tanh(mean)  # Constrain to [-1, 1]

#         # Prepare RNN states for output
#         rnn_states = [
#             new_hidden_states[0].view(batch_size, -1).contiguous(),
#             new_hidden_states[1].view(batch_size, -1).contiguous(),
#         ]

#         return mean, self.log_std_parameter, {"rnn": rnn_states}

#     def _get_zero_hidden_states(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
#         """Get zero-initialized hidden states."""
#         return (
#             torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=self.device),
#             torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=self.device),
#         )


class ValueLSTMNW(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        clip_actions: bool = False,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        num_envs: int = 1,
        sequence_length: int = 128,
    ) -> None:
        """Initialize the Critic model with LSTM.

        Args:
            observation_space: The observation space
            action_space: The action space
            device: The device to run the model on
            clip_actions: Whether to clip actions
            lstm_hidden_size: Hidden size for LSTM layer
            lstm_num_layers: Number of LSTM layers
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        input_size = int(calculate_flattened_obs_space_size(observation_space))
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.sequence_length = sequence_length
        self.num_envs = num_envs

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # Pre-LSTM layers
        self.pre_lstm = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Post-LSTM layers
        self.post_lstm = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Initialize hidden states
        self.hidden_states = None

    def get_specification(self) -> dict:
        """Get model specification including RNN information."""
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (
                        self.lstm_num_layers,
                        self.num_envs,
                        self.lstm_hidden_size,
                    ),  # Hidden states
                    (
                        self.lstm_num_layers,
                        self.num_envs,
                        self.lstm_hidden_size,
                    ),  # Cell states
                ],
            }
        }

    def compute(self, inputs: dict, role: str) -> tuple[torch.Tensor, dict]:
        """Compute the value function for the given inputs.

        Args:
            inputs: Dictionary containing states and potentially RNN states. States are already flattened.
            role: Role of the model

        Returns:
            Tuple containing the computed value and RNN states dict
        """
        x = inputs["states"]
        batch_size = x.shape[0]

        # Get RNN states from inputs or initialize
        if "rnn" in inputs and inputs["rnn"] is not None and len(inputs["rnn"]) >= 2:
            # Use provided RNN states - reshape from flattened form
            rnn_states = inputs["rnn"]
            hidden_states = (
                rnn_states[0].view(self.lstm_num_layers, batch_size, self.lstm_hidden_size),
                rnn_states[1].view(self.lstm_num_layers, batch_size, self.lstm_hidden_size),
            )
        else:
            # Initialize hidden states
            hidden_states = (
                torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=self.device),
                torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=self.device),
            )

        # Pre-LSTM processing
        x = self.pre_lstm(x)

        # Add sequence dimension for LSTM (batch_size, seq_len=1, features)
        x = x.unsqueeze(1)

        # LSTM forward pass
        lstm_out, new_hidden_states = self.lstm(x, hidden_states)

        # Remove sequence dimension
        lstm_out = lstm_out.squeeze(1)

        # Post-LSTM processing
        value = self.post_lstm(lstm_out)

        # Prepare RNN states for output
        # rnn_states = [
        #     new_hidden_states[0].view(batch_size, -1).contiguous(),  # flatten hidden state
        #     new_hidden_states[1].view(batch_size, -1).contiguous(),  # flatten cell state
        # ]

        rnn_states = [new_hidden_states[0], new_hidden_states[1]]

        return value, {"rnn": rnn_states}


# Configure PPO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Configuration: {CONFIG_PATH}")
print(f"Experiment name: {EXPERIMENT_NAME}")

# Models
models = {}
models["policy"] = PolicyNW(env.observation_space, env.action_space, device)
models["value"] = ValueLSTMNW(env.observation_space, env.action_space, device)

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
cfg["experiment"]["experiment_name"] = EXPERIMENT_NAME

# Memory
memory = RandomMemory(memory_size=cfg["rollouts"], num_envs=env.num_envs, device=device)

# Create agent
agent = PPO_RNN(
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

    cfg_trainer = {"timesteps": 300000, "headless": True}
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
