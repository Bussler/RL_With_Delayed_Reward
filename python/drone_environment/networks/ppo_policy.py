import gymnasium as gym
import torch
from drone_environment.gym import calculate_flattened_obs_space_size
from skrl.models.torch import GaussianMixin, Model
from torch import nn


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
        mean = torch.tanh(mean)  # TODO test if this is a problem!

        return mean, self.log_std_parameter, {"features": features}
