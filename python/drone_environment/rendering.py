import io
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

Renderer = Literal["matplotlib"]
RenderMode = Literal["human", "rgb_array"] | None


class BaseRenderer(ABC):
    """Base class for rendering the drone environment.

    Attributes:
        render_mode: either "human" to display the environment in a window,
            "rgb_array" to return an RGB array, or None
        coords_size: the size of the coordinate system to be displayed
        frame_cache: a list to cache rendered frames
    """

    def __init__(self, render_mode: RenderMode, coords_size: float):
        """Initializes the BaseRenderer."""
        self.render_mode = render_mode
        self.coords_size = coords_size
        self.frame_cache: list[np.ndarray[Any, Any]] = []

    def render_step(
        self,
        player_position: tuple[float, float, float],
        target_positions: dict[int, tuple[bool, tuple[float, float, float]]],
        time: float,
        additional_info: dict | None = None,
    ) -> np.ndarray | None:
        """Renders a frame of the drone environment to the specified render mode.

        Args:
            player_position: Current position of the player as (x, y, z)
            target_positions: Dictionary mapping target_id to (x, y, z) position
            time: Current simulation time
            additional_info: Optional dictionary with additional rendering info

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        frame = self._draw_frame(player_position, target_positions, time, additional_info)

        if frame is not None:
            self.frame_cache.append(frame)

        match self.render_mode:
            case "human":
                self._render_human()
                return None
            case "rgb_array":
                return self._render_rgb_array()
            case None:
                return None

    @abstractmethod
    def _draw_frame(
        self,
        player_position: tuple[float, float, float],
        target_positions: dict[int, tuple[bool, tuple[float, float, float]]],
        time: float,
        additional_info: dict | None = None,
    ) -> Any:
        """Draws a single frame of the environment."""

    @abstractmethod
    def _render_human(self) -> None:
        """Renders the environment to a human-readable format, such as a window."""

    @abstractmethod
    def _render_rgb_array(self) -> np.ndarray | None:
        """Renders the environment to an RGB array."""

    @abstractmethod
    def close(self) -> None:
        """Cleans up any resources used by the renderer."""


class MatPlotLibRenderer(BaseRenderer):
    def __init__(self, render_mode: RenderMode, coords_size: float, trail_length: int = 20):
        super().__init__(render_mode, coords_size)

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.output_dir = "animations"
        self.trail_length = trail_length

        # Track target colors for consistency across frames
        self.target_colors: dict[int, str] = {}
        self.color_palette = ["red", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

        # Store position histories for trails
        self.player_position_history: list[tuple[float, float, float]] = []
        self.target_position_history: dict[int, list[tuple[float, float, float]]] = {}

        # Information about hit targets or expired targets
        self.hit_targets = 0
        self.expired_targets = 0

    def _get_target_color(self, target_id: int) -> str:
        """Get a consistent color for a target ID"""
        if target_id not in self.target_colors:
            color_idx = len(self.target_colors) % len(self.color_palette)
            self.target_colors[target_id] = self.color_palette[color_idx]
        return self.target_colors[target_id]

    def _update_position_histories(
        self,
        player_position: tuple[float, float, float],
        target_positions: dict[int, tuple[bool, tuple[float, float, float]]],
    ) -> None:
        """Update position histories for trails"""
        # Update player position history
        self.player_position_history.append(player_position)
        if len(self.player_position_history) > self.trail_length:
            self.player_position_history.pop(0)

        # Update target position histories
        targets_to_remove = []
        for target_id, info in target_positions.items():
            is_dead = info[0]
            if is_dead:
                targets_to_remove.append(target_id)
                continue

            pos = info[1]
            if target_id not in self.target_position_history:
                self.target_position_history[target_id] = []

            self.target_position_history[target_id].append(pos)
            if len(self.target_position_history[target_id]) > self.trail_length:
                self.target_position_history[target_id].pop(0)

        # Clean up histories for removed targets
        for target_id in targets_to_remove:
            if target_id in self.target_position_history:
                del self.target_position_history[target_id]

    def _draw_trails(self) -> None:
        """Draw trails for player and targets"""
        # Draw player trail
        if len(self.player_position_history) > 1:
            player_trail = np.array(self.player_position_history)
            self.ax.plot(
                player_trail[:, 0],
                player_trail[:, 1],
                player_trail[:, 2],
                color="blue",
                alpha=0.5,
                linewidth=1,
                linestyle="-",
            )

        # Draw target trails
        for target_id, positions in self.target_position_history.items():
            if len(positions) > 1:
                color = self._get_target_color(target_id)
                target_trail = np.array(positions)
                self.ax.plot(
                    target_trail[:, 0],
                    target_trail[:, 1],
                    target_trail[:, 2],
                    color=color,
                    alpha=0.5,
                    linewidth=1,
                    linestyle="-",
                )

    def _draw_frame(
        self,
        player_position: tuple[float, float, float],
        target_positions: dict[int, tuple[bool, tuple[float, float, float]]],
        time: float,
        additional_info: dict | None = None,
    ) -> Any:
        # Update position histories for trails
        self._update_position_histories(player_position, target_positions)

        self.ax.clear()

        # Set the plot limits
        self.ax.set_xlim((-self.coords_size / 2, self.coords_size / 2))
        self.ax.set_ylim((-self.coords_size / 2, self.coords_size / 2))
        self.ax.set_zlim((-self.coords_size / 2, self.coords_size / 2))  # type: ignore

        # Set labels
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")  # type: ignore

        # Create title with additional info if available
        title = f"Drone Defense Simulation - Time: {time:.1f}s"
        if additional_info:
            if "hit_targets" in additional_info:
                self.hit_targets += additional_info["hit_targets"]
                title += f" | Hits: {self.hit_targets}"
            if "expired_targets" in additional_info:
                self.expired_targets += additional_info["expired_targets"]
                title += f" | Expired: {self.expired_targets}"

        self.ax.set_title(title)

        # Draw trails first (so they appear behind the markers)
        self._draw_trails()

        # Plot player position
        self.ax.scatter(
            player_position[0],
            player_position[1],
            player_position[2],
            color="blue",
            marker="o",
            s=100,  # type: ignore
            label="Player",
            edgecolors="black",
            linewidth=1,
            zorder=10,  # Ensure player marker is on top
        )

        # Plot targets
        active_targets = []
        for target_id, info in target_positions.items():
            is_dead = info[0]
            if is_dead:
                continue
            pos = info[1]
            color = self._get_target_color(target_id)
            self.ax.scatter(
                pos[0],
                pos[1],
                pos[2],
                color=color,
                marker="^",
                s=100,  # type: ignore
                label=f"Target {target_id}",
                edgecolors="black",
                linewidth=1,
                zorder=10,  # Ensure target markers are on top
            )
            active_targets.append(target_id)

        # Add legend if there are targets or player
        if target_positions or player_position:
            # Create custom legend entries to avoid duplicate trail entries
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="blue",
                    markerfacecolor="blue",
                    markersize=10,
                    label="Player",
                    linestyle="None",
                )
            ]

            for target_id in active_targets:
                color = self._get_target_color(target_id)
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="^",
                        color=color,
                        markerfacecolor=color,
                        markersize=8,
                        label=f"Target {target_id}",
                        linestyle="None",
                    )
                )

            self.ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

        # Draw and cache the frame
        self.fig.canvas.draw()

        # Convert plot to image and cache it
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)

        # Convert to RGB array for consistency
        img_array = np.array(img)
        buf.close()

        return img_array

    def _render_human(self) -> None:
        plt.pause(0.01)
        plt.show(block=False)

    def _render_rgb_array(self) -> np.ndarray | None:
        """Return the most recent frame as RGB array."""
        if self.frame_cache:
            return self.frame_cache[-1]
        return None

    def reset_trails(self) -> None:
        """Reset all position histories for trails"""
        self.player_position_history.clear()
        self.target_position_history.clear()

    def set_trail_length(self, length: int) -> None:
        """Set the length of trails to display"""
        self.trail_length = max(1, length)

        # Trim existing histories if they're longer than new length
        if len(self.player_position_history) > self.trail_length:
            self.player_position_history = self.player_position_history[-self.trail_length :]

        for target_id in self.target_position_history:
            if len(self.target_position_history[target_id]) > self.trail_length:
                self.target_position_history[target_id] = self.target_position_history[target_id][
                    -self.trail_length :
                ]

    def close(self) -> None:
        """Save animation and close matplotlib figure."""
        if len(self.frame_cache) > 0:
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = os.path.join(self.output_dir, f"drone_simulation_{timestamp}.gif")

            try:
                # Convert numpy arrays back to PIL Images if needed
                pil_frames = []
                for frame in self.frame_cache:
                    if isinstance(frame, np.ndarray):
                        pil_frames.append(Image.fromarray(frame))
                    else:
                        pil_frames.append(frame)  # type: ignore

                if pil_frames:
                    pil_frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=pil_frames[1:],
                        optimize=False,
                        duration=100,  # milliseconds per frame
                        loop=0,  # loop forever
                    )
                    print(f"Animation saved to {gif_path}")

            except Exception as e:
                print(f"Failed to create GIF animation: {e}")

        plt.close(self.fig)

    def save_current_frame(self, filename: str | None = None) -> str:
        """Save the current frame as a static image"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drone_frame_{timestamp}.png"

        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        self.fig.savefig(filepath, dpi=150, bbox_inches="tight")
        return filepath


def get_renderer(renderer: Renderer, render_mode: RenderMode, coords_size: float) -> BaseRenderer:
    """Create a renderer instance."""
    match renderer:
        case "matplotlib":
            cls = MatPlotLibRenderer
        case _:
            raise ValueError(f"Renderer {renderer} is unknown.")
    return cls(render_mode, coords_size)
