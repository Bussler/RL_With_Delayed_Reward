from typing import Any

import gymnasium as gym
import numpy as np
from drone_environment._lib import DoneEnvironmentWrapper
from drone_environment.rendering import Renderer, RenderMode, get_renderer
from drone_environment.utils import add_random_targets, read_yml
from gymnasium import spaces


class DroneGymEnv(gym.Env):
    """Gymnasium wrapper for the Rust DroneEnvironment using DoneEnvironmentWrapper.

    This environment simulates a drone defense scenario where a player
    needs to intercept targets (enemy drones) by moving within collision range.
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_time: float = 100.0,
        player_speed: float = 8.0,
        collision_radius: float = 0.5,
        num_targets: int = 5,
        max_target_speed: float = 6.0,
        max_flight_time_range: tuple[float, float] = (5.0, 15.0),
        arena_size: float = 50.0,
        renderer: Renderer | None = None,
        render_mode: RenderMode | None = None,
    ) -> None:
        """Initialize the drone environment.

        Args:
            dt: Time step size in seconds for simulation updates. Smaller values provide
                higher precision but slower simulation.
            max_time: Maximum simulation time in seconds before episode termination.
            player_speed: Movement speed of the player units in units per second.
            collision_radius: Detection radius in units for player-target collisions.
                Player within this distance of targets are considered hits.
            num_targets: Number of enemy targets to spawn in the environment.
            max_target_speed: Maximum speed in units per second that targets can move.
            max_flight_time_range: Tuple of (min_time, max_time) in seconds defining the
                range of flight times for targets before they expire.
            arena_size: Side length of the cubic simulation arena in units. Entities
                operate within a cube of dimensions arena_size × arena_size × arena_size.
            renderer: Optional renderer type for visualization. If None, no rendering
                is performed. Currently supports "matplotlib".
            render_mode: Rendering mode for visualization. Options include "human" for
                interactive display or "rgb_array" for programmatic access to frames.
        """
        self.dt = dt
        self.max_time = max_time
        self.player_speed = player_speed
        self.collision_radius = collision_radius
        self.num_targets = num_targets
        self.max_target_speed = max_target_speed
        self.max_flight_time_range = max_flight_time_range
        self.arena_size = arena_size

        self.render_mode = render_mode
        self.renderer = (
            None if renderer is None else get_renderer(renderer, self.render_mode, self.arena_size)
        )

        # Create the Rust environment using DoneEnvironmentWrapper
        self.env = DoneEnvironmentWrapper(
            player_position=(0.0, 0.0, 0.0),
            player_speed=self.player_speed,
            dt=self.dt,
            max_time=self.max_time,
            collision_radius=self.collision_radius,
        )

        add_random_targets(
            self.env, self.num_targets, arena_size, max_target_speed, 0.6, max_flight_time_range
        )

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

        # TODO add variation: create reset with different scenario?

        self.env.reset(player_position=(0.0, 0.0, 0.0))

        # Get the initial observation
        observation_raw = self.env.get_observation()
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
                    low=-self.arena_size,
                    high=self.arena_size,
                    shape=(3,),
                    dtype=np.float64,
                ),
                # Target IDs: [num_targets]
                "target_ids": spaces.Box(
                    low=0,
                    high=self.num_targets - 1,
                    shape=(self.num_targets,),
                    dtype=np.int8,
                ),
                # Target positions: [num_targets, 3]
                "target_positions": spaces.Box(
                    low=-self.arena_size,
                    high=self.arena_size,
                    shape=(self.num_targets, 3),
                    dtype=np.float64,
                ),
                # Target velocities: [num_targets, 3]
                "target_velocities": spaces.Box(
                    low=-self.max_target_speed,
                    high=self.max_target_speed,
                    shape=(self.num_targets, 3),
                    dtype=np.float64,
                ),
                # Target distances: [num_targets]
                "target_distances": spaces.Box(
                    low=0.0,
                    high=self.arena_size * 2,  # Maximum possible distance in arena
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
    test_env = DroneGymEnv()

    print(f"Observation space: {calculate_flattened_obs_space_size(test_env.observation_space)}")
    print(f"Action space: {test_env.action_space}")

    action = test_env.action_space.sample()
    obs, info = test_env.reset()

    print(f"Sample action: {action}")
    print(f"Initial observation keys: {obs.keys()}")
    print(f"Player position: {obs['player_position']}")
    print(f"Active targets: {np.sum(obs['target_mask'])}")

    observation, reward, done, truncated, info = test_env.step(action)
    print(f"Step reward: {reward}, done: {done}")
