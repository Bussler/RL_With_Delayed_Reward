import random
from typing import Any

import yaml


def _round_to_one_decimal(value: float) -> float:
    """Round a float value to one decimal place."""
    return round(value, 1)


def generate_random_target(
    target_id: int,
    arena_size: float = 20.0,
    max_speed: float = 6.0,
    trajectory_probability: float = 0.6,
    max_flight_time_range: tuple[float, float] = (5.0, 15.0),
) -> dict:
    """Generate a random target with either velocity-based linear motion or trajectory function.

    Args:
        target_id: Unique identifier for the target
        arena_size: Size of the arena to spawn targets within
        max_speed: Maximum speed for linear motion targets
        trajectory_probability: Probability of generating a trajectory-based target (0.0-1.0)
        max_flight_time_range: Range for random max_flight_time (min, max)

    Returns:
        Dictionary with target parameters for env.add_target()
    """
    # Random starting position
    position = (
        _round_to_one_decimal(random.uniform(-arena_size, arena_size)),
        _round_to_one_decimal(random.uniform(-arena_size, arena_size)),
        _round_to_one_decimal(random.uniform(1.0, 10.0)),  # Height between 1-10 units
    )

    # Decide between linear motion (velocity) or trajectory function
    use_trajectory = random.random() < trajectory_probability

    if use_trajectory:
        # Generate trajectory-based target
        trajectory_fn = _generate_random_trajectory()
        velocity = (0.0, 0.0, 0.0)  # Not used for trajectory targets
        max_flight_time = _round_to_one_decimal(random.uniform(*max_flight_time_range))
    else:
        # Generate linear motion target
        trajectory_fn = None
        velocity = (
            _round_to_one_decimal(random.uniform(-max_speed, max_speed)),
            _round_to_one_decimal(random.uniform(-max_speed, max_speed)),
            _round_to_one_decimal(random.uniform(-max_speed / 2, max_speed / 2)),  # Slower vertical movement
        )
        max_flight_time = None  # Linear targets don't need flight time limit

    return {
        "target_id": target_id,
        "position": position,
        "velocity": velocity,
        "trajectory_fn": trajectory_fn,
        "max_flight_time": max_flight_time,
    }


def _generate_random_trajectory() -> str:
    """Generate a random trajectory function string."""
    trajectory_types = [
        "circular_xy",
        "circular_xz",
        "circular_yz",
        "spiral",
        "figure_eight",
        "sine_wave",
        "helical",
    ]

    trajectory_type = random.choice(trajectory_types)

    # Random parameters for trajectories
    radius = _round_to_one_decimal(random.uniform(3.0, 12.0))
    frequency = _round_to_one_decimal(random.uniform(0.5, 2.0))
    amplitude = _round_to_one_decimal(random.uniform(2.0, 8.0))
    center_x = _round_to_one_decimal(random.uniform(-5.0, 5.0))
    center_y = _round_to_one_decimal(random.uniform(-5.0, 5.0))
    center_z = _round_to_one_decimal(random.uniform(3.0, 8.0))

    if trajectory_type == "circular_xy":
        return (
            f"{center_x} + {radius}*cos({frequency}*t), {center_y} + {radius}*sin({frequency}*t), {center_z}"
        )

    if trajectory_type == "circular_xz":
        return (
            f"{center_x} + {radius}*cos({frequency}*t), {center_y}, {center_z} + {radius}*sin({frequency}*t)"
        )

    if trajectory_type == "circular_yz":
        return (
            f"{center_x}, {center_y} + {radius}*cos({frequency}*t), {center_z} + {radius}*sin({frequency}*t)"
        )

    if trajectory_type == "spiral":
        spiral_rate = _round_to_one_decimal(random.uniform(0.1, 0.5))
        return f"{center_x} + {spiral_rate}*t*cos({frequency}*t), {center_y} + {spiral_rate}*t*sin({frequency}*t), {center_z}"

    if trajectory_type == "figure_eight":
        return f"{center_x} + {radius}*sin({frequency}*t), {center_y} + {radius}*sin({frequency}*t)*cos({frequency}*t), {center_z}"

    if trajectory_type == "sine_wave":
        direction = random.choice(["x", "y", "z"])
        if direction == "x":
            return f"{center_x} + {amplitude}*sin({frequency}*t), {center_y} + t, {center_z}"
        if direction == "y":
            return f"{center_x} + t, {center_y} + {amplitude}*sin({frequency}*t), {center_z}"
        # z direction
        return f"{center_x} + t, {center_y}, {center_z} + {amplitude}*sin({frequency}*t)"

    if trajectory_type == "helical":
        vertical_speed = _round_to_one_decimal(random.uniform(0.5, 2.0))
        return f"{center_x} + {radius}*cos({frequency}*t), {center_y} + {radius}*sin({frequency}*t), {center_z} + {vertical_speed}*t"

    # Fallback to simple circular
    return f"{center_x} + {radius}*cos({frequency}*t), {center_y} + {radius}*sin({frequency}*t), {center_z}"


def create_environment_config(
    num_targets: int = 5,
    arena_size: float = 50.0,
    max_speed: float = 6.0,
    trajectory_probability: float = 0.6,
    max_flight_time_range: tuple[float, float] = (10.0, 30.0),
    output_file: str = "generated_config.yaml",
) -> dict:
    """Create a complete environment configuration with random targets.

    Args:
        num_targets: Number of random targets to generate
        arena_size: Size of the arena for target placement
        max_speed: Maximum speed for linear motion targets
        trajectory_probability: Probability of generating trajectory-based targets
        max_flight_time_range: Range for target flight times
        output_file: Output YAML file path

    Returns:
        Dictionary containing the complete configuration
    """
    # Base configuration structure
    config: dict[str, Any] = {
        "player": {"position": [0.0, 0.0, 0.0], "speed": 8.0},
        "environment": {"dt": 0.1, "max_time": 100.0, "collision_radius": 0.5},
        "targets": [],
    }

    # Generate random targets
    print(f"Generating {num_targets} random targets...")
    for i in range(num_targets):
        target_params = generate_random_target(
            target_id=i + 1,  # Start IDs from 1
            arena_size=arena_size,
            max_speed=max_speed,
            trajectory_probability=trajectory_probability,
            max_flight_time_range=max_flight_time_range,
        )

        # Convert to config format
        target_config = {
            "id": target_params["target_id"],
            "position": list(target_params["position"]),
            "velocity": list(target_params["velocity"]),
            "trajectory_fn": target_params["trajectory_fn"],
            "max_flight_time": target_params["max_flight_time"],
        }

        config["targets"].append(target_config)

        # Print target info
        if target_params["trajectory_fn"]:
            print(f"  Target {target_params['target_id']}: Trajectory - {target_params['trajectory_fn']}")
        else:
            print(
                f"  Target {target_params['target_id']}: Linear motion - velocity={target_params['velocity']}"
            )

    # Save to YAML file
    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"\nConfiguration saved to: {output_file}")
    return config


def main() -> None:
    """Main function to generate and display the configuration."""
    print("=== Drone Environment Config Generator ===\n")

    # Generate configuration with custom parameters
    config = create_environment_config(
        num_targets=3,
        arena_size=25.0,
        max_speed=5.0,
        trajectory_probability=0.7,
        max_flight_time_range=(10.0, 30.0),
        output_file="configs/drone_env/generated_config.yaml",
    )

    print("\n=== Generated Configuration ===")
    print(yaml.dump(config, default_flow_style=False, indent=2))


if __name__ == "__main__":
    main()
