import drone_environment as de
import matplotlib.pyplot as plt
import numpy as np
from drone_environment.utils import add_random_targets
from matplotlib.animation import FuncAnimation

# Create environment
env = de.DoneEnvironmentWrapper(
    player_position=(0.0, 0.0, 0.0), player_speed=8.0, dt=0.1, max_time=200.0, collision_radius=0.5
)

# Add targets with different trajectories

# Linear motion

add_random_targets(env, num_targets=3)

# Reset environment and get initial observation
obs = env.reset((0.0, 0.0, 0.0))

# Number of simulation steps
num_steps = 200

# Storage for positions - use dictionaries keyed by target_id to handle target removal
player_positions = []
target_positions_by_frame = []  # List of dicts: [{target_id: (x,y,z), ...}, ...]

# Run simulation and record positions
for i in range(num_steps):
    # Get current positions
    player_pos = env.get_player_position()
    current_targets = env.get_target_positions()

    player_positions.append((player_pos[0], player_pos[1], player_pos[2]))

    # Store target positions for this frame using target IDs
    frame_targets = {}
    direction = None
    for target_id, target_position in current_targets.items():
        target_pos = (target_position[0], target_position[1], target_position[2])
        frame_targets[target_id] = target_pos

        if direction is None:
            direction = (
                target_position[0] - player_pos[0],
                target_position[1] - player_pos[1],
                target_position[2] - player_pos[2],
            )

    if direction is None:
        direction = (0.0, 0.0, 0.0)

    target_positions_by_frame.append(frame_targets)

    # Take a step in the environment
    obs, reward, done, info = env.step(direction)
    if done:
        break

# Convert player positions to numpy array
player_positions = np.array(player_positions)

# Find the overall min and max ranges for consistent plotting
all_positions = []
for frame_targets in target_positions_by_frame:
    all_positions.extend(frame_targets.values())
all_positions.extend([(p[0], p[1], p[2]) for p in player_positions])

if all_positions:
    all_positions = np.array(all_positions)
    max_range = np.max(all_positions)
    min_range = np.min(all_positions)
else:
    max_range, min_range = 10, -10

# Create figure for animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Color map for different targets
colors = ["red", "green", "orange", "purple", "brown", "pink", "gray", "olive"]


# Animation update function
def update(frame: int) -> None:
    ax.clear()

    # Plot player
    ax.scatter(
        player_positions[frame, 0],
        player_positions[frame, 1],
        player_positions[frame, 2],
        color="blue",
        marker="o",
        s=100,
        label="Player",
    )

    # Plot active targets for this frame
    frame_targets = target_positions_by_frame[frame]
    for target_id, target_pos in frame_targets.items():
        color = colors[target_id % len(colors)]

        ax.scatter(
            target_pos[0],
            target_pos[1],
            target_pos[2],
            color=color,
            marker="o",
            s=100,
            label=f"Target {target_id}",
        )

        # Build trail for this target by looking back through previous frames
        trail_positions = []
        start_frame = max(0, frame - 19)  # Last 20 positions including current

        for f in range(start_frame, frame + 1):
            if f < len(target_positions_by_frame) and target_id in target_positions_by_frame[f]:
                trail_positions.append(target_positions_by_frame[f][target_id])

        # Plot trail if we have more than one position
        if len(trail_positions) > 1:
            trail_array = np.array(trail_positions)
            ax.plot(
                trail_array[:, 0],
                trail_array[:, 1],
                trail_array[:, 2],
                color=color,
                alpha=0.5,
            )

    # Plot player trail (last 20 positions)
    start_idx = max(0, frame - 19)
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
    ax.set_title(f"Drone Environment Simulation - Frame {frame}\nActive Targets: {len(frame_targets)}")
    ax.legend()


# Create animation
ani = FuncAnimation(fig, update, frames=min(num_steps, len(player_positions)), interval=100, blit=False)

# Save as GIF
ani.save("drone_simulation.gif", writer="pillow", fps=10)

print("Animation saved as 'drone_simulation.gif'")
