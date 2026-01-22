from collections.abc import Sequence

import gymnasium as gym
import torch
from torch import nn

from drone_environment.gym import calculate_flattened_obs_space_size
from skrl.models.torch import DeterministicMixin, Model


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


class ValueLSTM(Model):
    """Recurrent (LSTM) value network."""

    num_observations: int

    def __init__(
        self,
        observation_space: int | Sequence[int] | gym.Space,
        action_space: int | Sequence[int] | gym.Space,
        device: str | torch.device | None = None,
        num_envs: int = 1,
        sequence_length: int = 128,  # Time Horizon we want the LSTM to remember
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
    ) -> None:
        """Initialize a value network with lstm for temporal context."""
        super().__init__(observation_space, action_space, device)

        self.sequence_length = sequence_length
        self.batch_size = num_envs
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # self.value_head = nn.Linear(lstm_hidden_size, 1)
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_specification(self) -> dict:
        """Setup initial lstm states."""
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (
                        self.lstm_num_layers,
                        self.batch_size,
                        self.lstm_hidden_size,
                    ),  # hidden states (D num_layers, N, Hout)
                    (self.lstm_num_layers, self.batch_size, self.lstm_hidden_size),
                ],
            }
        }  # cell states   (D num_layers, N, Hcell)

    def _forward_core(
        self,
        states: torch.Tensor,
        rnn_states_in: torch.Tensor,
        terminated: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden_states, cell_states = rnn_states_in[0], rnn_states_in[1]

        x = self.encoder(states)

        # training: reset cell, hidden state at termination of sequence
        if self.training:
            B, obs_dim = x.shape  # noqa: N806

            rnn_input = x.view(
                -1,
                B,
                obs_dim,
            )
            hidden_states = hidden_states.view(
                self.lstm_num_layers, -1, B, hidden_states.shape[-1]
            )  # (num_layers, 1, B, Hout)
            cell_states = cell_states.view(
                self.lstm_num_layers, -1, B, cell_states.shape[-1]
            )  # (num_layers, 1, B, Hcell)

            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
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
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = x.view(-1, 1, x.shape[-1])  # (B, 1, Hin)
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        values = self.value_head(rnn_output)

        return values, rnn_states

    def act(
        self,
        inputs: dict,
        role: str = "",  # noqa: ARG002
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, dict]:
        """Forward inputs through the network."""
        states = inputs["states"]
        rnn_states = inputs.get("rnn")
        terminated = inputs.get("terminated")

        if rnn_states is None:
            message = "RNN States not instantiated!"
            raise ValueError(message)

        values, new_rnn_states = self._forward_core(states, rnn_states, terminated)
        info = {"rnn": new_rnn_states}
        return values, new_rnn_states, info
