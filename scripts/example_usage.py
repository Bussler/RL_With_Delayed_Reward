import drone_environment as de
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Create environment
env = de.DroneEnvironment(
    player_position=de.Vec3(0.0, 0.0, 0.0), player_speed=8.0, dt=0.1, max_time=200.0, collision_radius=0.5
)

# Add targets with different trajectories

# Linear motion
env.add_target(
    target_id=0,
    position=de.Vec3(-10.0, -10.0, 3.0),
    velocity=de.Vec3(1.0, 1.0, 0.0),
    trajectory_fn=None,  # None/ Empty string means use velocity for linear motion
    max_flight_time=None,
)

# Circular motion in XY plane
env.add_target(
    target_id=1,
    position=de.Vec3(10.0, 0.0, 5.0),
    velocity=de.Vec3(0.0, 0.0, 0.0),
    trajectory_fn="7*cos(t), 7*sin(t), 5.0",
    max_flight_time=10,
)

# Reset environment and get initial observation
obs = env.reset(de.Vec3(0.0, 0.0, 0.0))

# Number of simulation steps
num_steps = 200

# Storage for positions - use lists of lists to handle varying number of targets
player_positions = []
target_positions = []
active_targets = []  # Track which targets are active in each frame

# Run simulation and record positions
for i in range(num_steps):
    # Get current positions
    player_pos = env.get_player_position()
    current_targets = env.get_target_positions()

    player_positions.append((player_pos.x, player_pos.y, player_pos.z))

    # Store target positions for this frame
    # Example action: move in the direction of the first available target
    frame_targets = []
    direction = None
    for _, target_position in current_targets:
        frame_targets.append((target_position.x, target_position.y, target_position.z))

        if direction is None:
            direction = de.Vec3(
                target_position.x - player_pos.x,
                target_position.y - player_pos.y,
                target_position.z - player_pos.z,
            )

    if direction is None:
        direction = de.Vec3(0.0, 0.0, 0.0)

    target_positions.append(frame_targets)
    active_targets.append(len(current_targets))

    # Take a step in the environment
    obs, reward, done, info = env.step(direction)
    if done:
        break

# Convert player positions to numpy array
player_positions = np.array(player_positions)

# Find the overall min and max ranges for consistent plotting
all_positions = [pos for frame_targets in target_positions for pos in frame_targets]
all_positions.extend([(p[0], p[1], p[2]) for p in player_positions])
# all_positions = np.array(all_positions) if all_positions else np.array([[0, 0, 0]])

max_range = np.max(all_positions, axis=0).max()
min_range = np.min(all_positions, axis=0).min()

# Create figure for animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Target colors for consistency
target_colors = [
    "red",
    "green",
    "orange",
    "purple",
    "cyan",
]  # Add more colors if needed


# Animation update function
def update(frame: int) -> None:
    ax.clear()

    # Plot player
    ax.scatter(
        player_positions[frame, 0],
        player_positions[frame, 1],
        player_positions[frame, 2],
        color="blue",
        s=100,
        label="Player",
    )

    # Plot active targets for this frame
    frame_targets = target_positions[frame]
    for i, target_pos in enumerate(frame_targets):
        color = target_colors[i % len(target_colors)]
        ax.scatter(
            target_pos[0],
            target_pos[1],
            target_pos[2],
            color=color,
            s=100,
            label=f"Target {i + 1}",
        )

        # Plot trails for this target (last 20 positions)
        start_idx = max(0, frame - 20)
        trail_positions = []
        for f in range(start_idx, frame + 1):
            if f < len(target_positions) and i < len(target_positions[f]):
                trail_positions.append(target_positions[f][i])

        if trail_positions:
            trail_positions = np.array(trail_positions)
            ax.plot(
                trail_positions[:, 0],
                trail_positions[:, 1],
                trail_positions[:, 2],
                color=color,
                alpha=0.5,
            )

    # Plot player trail (last 20 positions)
    start_idx = max(0, frame - 20)
    ax.plot(
        player_positions[start_idx : frame + 1, 0],
        player_positions[start_idx : frame + 1, 1],
        player_positions[start_idx : frame + 1, 2],
        "b-",
        alpha=0.5,
    )

    # Set consistent axis limits
    buffer = 2.0  # Add some buffer around the min/max values
    ax.set_xlim(min_range - buffer, max_range + buffer)
    ax.set_ylim(min_range - buffer, max_range + buffer)
    ax.set_zlim(min_range - buffer, max_range + buffer)

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Drone Environment Simulation - Frame {frame}\nActive Targets: {active_targets[frame]}")
    ax.legend()


# Create animation
ani = FuncAnimation(fig, update, frames=min(num_steps, len(player_positions)), interval=100, blit=False)

# Save as GIF
ani.save("drone_simulation.gif", writer="pillow", fps=10)

print("Animation saved as 'drone_simulation.gif'")
