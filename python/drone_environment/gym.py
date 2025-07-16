from typing import Any

import gymnasium as gym
import numpy as np
from drone_environment._lib import DroneEnvironmentWrapper
from drone_environment.rendering import Renderer, RenderMode, get_renderer
from drone_environment.utils import read_yml
from gymnasium import spaces


class DroneGymEnv(gym.Env):
    """Gymnasium wrapper for the Rust DroneEnvironment using DroneEnvironmentWrapper.

    This environment simulates a drone defense scenario where a player
    needs to intercept targets (enemy drones) by moving within collision range.
    """

    def __init__(
        self,
        drone_env_config: str = "configs/drone_env/default_config.yaml",
        renderer: Renderer | None = None,
        render_mode: RenderMode | None = None,
    ) -> None:
        """Initialize the drone environment.

        Args:
            drone_env_config: Path to the YAML configuration file for the DroneEnvironment.
            renderer: Optional renderer type for visualization. If None, no rendering
                is performed. Currently supports "matplotlib".
            render_mode: Rendering mode for visualization. Options include "human" for
                interactive display or "rgb_array" for programmatic access to frames.
        """
        # Create the Rust environment using DroneEnvironmentWrapper
        self.env = DroneEnvironmentWrapper.from_yaml_config(drone_env_config)
        _ = self.env.reset((0.0, 0.0, 0.0))

        self.arena_size = self.env.get_arena_size()

        self.render_mode = render_mode
        self.renderer = (
            None if renderer is None else get_renderer(renderer, self.render_mode, self.arena_size)
        )

        self.max_time = self.env.get_max_time()

        self.num_targets = len(self.env.get_target_positions())

        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()

    @classmethod
    def from_config(cls, f_path: str = "gym_config.yml") -> "DroneGymEnv":
        """Return an instance of the environment from a config file."""
        config = read_yml(f_path)
        return cls(**config)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment to an initial state."""
        if seed is not None:
            np.random.seed(seed)

        observation_raw = self.env.reset(player_position=(0.0, 0.0, 0.0))

        observation = self._process_observation(observation_raw)
        info: dict[str, Any] = self.env.get_information()

        return observation, info

    def step(self, action: np.ndarray[Any, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Array of movement direction [dx, dy, dz] normalized between -1 and 1

        Returns:
            observation: The current state
            reward: The reward for this step
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Convert normalized action to movement direction
        movement_action = (float(action[0]), float(action[1]), float(action[2]))

        # Take a step in the Rust environment
        observation_raw, reward, done, info_raw = self.env.step(movement_action)

        # Process the observation
        observation = self._process_observation(observation_raw)

        return observation, reward, done, False, info_raw

    def _create_action_space(self) -> spaces.Box:
        """Action space: 3D movement direction for the player.

        The action is a 3D vector (dx, dy, dz) with values between -1 and 1,
        representing the normalized movement direction.

        Returns:
            spaces.Box: Action space for 3D movement
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _create_observation_space(self) -> spaces.Dict:
        """Observation space: State representation matching Rust simulation output.

        Returns:
            spaces.Dict: Observation space matching Rust Observation struct
        """
        return spaces.Dict(
            {
                # Player position: [3] (x, y, z)
                "player_position": spaces.Box(
                    low=-self.arena_size * 3,
                    high=self.arena_size * 3,
                    shape=(3,),
                    dtype=np.float64,
                ),
                # Target positions: [num_targets, 3]
                "target_positions": spaces.Box(
                    low=-self.arena_size * 3,
                    high=self.arena_size * 3,
                    shape=(self.num_targets, 3),
                    dtype=np.float64,
                ),
                # Target velocities: [num_targets, 3]
                "target_velocities": spaces.Box(
                    low=-100,
                    high=100,
                    shape=(self.num_targets, 3),
                    dtype=np.float64,
                ),
                # Target distances: [num_targets]
                "target_distances": spaces.Box(
                    low=0.0,
                    high=self.arena_size * 3,  # Maximum possible distance in arena
                    shape=(self.num_targets,),
                    dtype=np.float64,
                ),
                "target_time_remaining": spaces.Box(
                    low=0.0,
                    high=self.max_time,
                    shape=(self.num_targets,),
                    dtype=np.float64,
                ),
                # Target mask (1 if target exists, 0 if not): [num_targets]
                "target_death_mask": spaces.Box(low=0, high=1, shape=(self.num_targets,), dtype=np.int8),
                # Time left: scalar
                "time_left": spaces.Box(low=0, high=self.max_time, shape=(1,), dtype=np.float64),
            }
        )

    def _process_observation(self, observation_raw: dict) -> dict[str, np.ndarray]:
        """Process raw observation from Rust into gym observation format.

        No need for processing for now, just return the raw observation.
        The rust code is already returning the correct format in numpy.

        Args:
            observation_raw: Raw observation dictionary from Rust simulation

        Returns:
            Processed observation matching the observation space
        """
        # No need for processing for now, just return the raw observation.

        return observation_raw

    def render(self) -> None:
        """Render the environment and cache the frames for later GIF creation."""
        if self.renderer is None:
            return

        player_position = self.env.get_player_position()
        target_positions = self.env.get_target_positions()
        info: dict[str, Any] = self.env.get_information()
        time = 0.0
        if "time" in info:
            time = info["time"]

        self.renderer.render_step(player_position, target_positions, time, info)

    def close(self) -> None:
        """Close the environment and create a GIF from cached frames if available."""
        if self.renderer is not None:
            self.renderer.close()

        return super().close()


def calculate_flattened_obs_space_size(obs_space: spaces.Space) -> int:
    """Calculate the total size of a flattened observation space."""
    if isinstance(obs_space, spaces.Dict):
        # For Dict spaces, sum up the sizes of all subspaces
        return sum(calculate_flattened_obs_space_size(subspace) for subspace in obs_space.values())
    if isinstance(obs_space, spaces.Box):
        # For Box spaces, multiply all dimensions
        return np.prod(obs_space.shape)  # type: ignore
    if isinstance(obs_space, spaces.Discrete):
        # For Discrete spaces, it's just 1
        return 1
    if isinstance(obs_space, spaces.MultiDiscrete):
        # For MultiDiscrete spaces, it's the sum of all discrete spaces
        return len(obs_space.nvec)
    raise ValueError(f"Unsupported space type: {type(obs_space)}")


if __name__ == "__main__":
    test_env = DroneGymEnv(renderer="matplotlib", render_mode="rgb_array")

    print(f"Observation space: {calculate_flattened_obs_space_size(test_env.observation_space)}")
    print(f"Action space: {test_env.action_space}")

    action = test_env.action_space.sample()
    obs, info = test_env.reset()

    print(f"Sample action: {action}")
    print(f"Active targets: {np.sum(obs['target_death_mask'])}")  # count amount of ones

    for i in range(200):
        action = test_env.action_space.sample()
        observation, reward, done, truncated, info = test_env.step(action)
        test_env.render()
        print(f"Step {i} reward: {reward}, done: {done}")
    test_env.close()
