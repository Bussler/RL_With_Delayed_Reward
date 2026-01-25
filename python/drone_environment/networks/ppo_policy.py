from collections.abc import Sequence

import gymnasium as gym
import torch
from torch import nn

from drone_environment.gym import calculate_flattened_obs_space_size
from skrl.models.torch import GaussianMixin, Model


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

        # network architecture
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
        mean = torch.tanh(mean)

        return mean, self.log_std_parameter, {"features": features}


class PolicyLSTM(GaussianMixin, Model):
    """Recurrent (LSTM) policy network producing Gaussian actions."""

    num_observations: int
    num_actions: int

    def __init__(
        self,
        observation_space: int | Sequence[int] | gym.Space,
        action_space: int | Sequence[int] | gym.Space,
        device: str | torch.device | None = None,
        *,
        clip_actions: bool = False,
        num_envs: int = 1,
        sequence_length: int = 128,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions)

        self.sequence_length = sequence_length
        self.batch_size = num_envs
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.mean_head = nn.Linear(lstm_hidden_size, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self) -> dict:
        """Setup initial LSTM states."""
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (
                        self.lstm_num_layers,
                        self.batch_size,
                        self.lstm_hidden_size,
                    ),
                    (
                        self.lstm_num_layers,
                        self.batch_size,
                        self.lstm_hidden_size,
                    ),
                ],
            }
        }

    def _forward_core(
        self,
        states: torch.Tensor,
        rnn_states_in: torch.Tensor,
        terminated: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden_states, cell_states = rnn_states_in[0], rnn_states_in[1]

        x = self.encoder(states)

        if self.training:
            B, obs_dim = x.shape

            rnn_input = x.view(-1, B, obs_dim)
            hidden_states = hidden_states.view(self.lstm_num_layers, -1, B, hidden_states.shape[-1])
            cell_states = cell_states.view(self.lstm_num_layers, -1, B, cell_states.shape[-1])

            hidden_states = hidden_states[:, :, 0, :].contiguous()
            cell_states = cell_states[:, :, 0, :].contiguous()

            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, B)
                indexes = [
                    0,
                    *(terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0].add(1).tolist()),
                    B,
                ]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(
                        rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                    )
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        else:
            rnn_input = x.view(-1, 1, x.shape[-1])
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        mean = self.mean_head(rnn_output)
        mean = torch.tanh(mean)

        return mean, self.log_std_parameter, rnn_states

    def compute(
        self,
        inputs: dict,
        role: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute the policy distribution parameters."""
        states = inputs["states"]
        rnn_states = inputs.get("rnn")
        terminated = inputs.get("terminated")

        if rnn_states is None:
            message = "RNN States not instantiated!"
            raise ValueError(message)

        mean, log_std, new_rnn_states = self._forward_core(states, rnn_states, terminated)
        info = {"rnn": new_rnn_states}
        return mean, log_std, info
