import gymnasium as gym
import torch
from drone_environment.gym import calculate_flattened_obs_space_size
from skrl.models.torch import DeterministicMixin, Model
from torch import nn


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


class ValueLSTM(DeterministicMixin, Model):
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
