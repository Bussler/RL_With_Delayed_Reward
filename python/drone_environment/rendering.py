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
        self.frame_cache = []

    def render_step(
        self,
        player_position: tuple[float, float, float],
        target_positions: dict[int, tuple[float, float, float]],
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
        target_positions: dict[int, tuple[float, float, float]],
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
    def __init__(self, render_mode: RenderMode, coords_size: float):
        super().__init__(render_mode, coords_size)

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.output_dir = "animations"

        # Track target colors for consistency across frames
        self.target_colors = {}
        self.color_palette = ["red", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    def _get_target_color(self, target_id: int) -> str:
        """Get a consistent color for a target ID"""
        if target_id not in self.target_colors:
            color_idx = len(self.target_colors) % len(self.color_palette)
            self.target_colors[target_id] = self.color_palette[color_idx]
        return self.target_colors[target_id]

    def _draw_frame(
        self,
        player_position: tuple[float, float, float],
        target_positions: dict[int, tuple[float, float, float]],
        time: float,
        additional_info: dict | None = None,
    ) -> Any:
        self.ax.clear()

        # Set the plot limits
        self.ax.set_xlim((-self.coords_size / 2, self.coords_size / 2))
        self.ax.set_ylim((-self.coords_size / 2, self.coords_size / 2))
        self.ax.set_zlim((0, self.coords_size))  # type: ignore

        # Set labels
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")  # type: ignore

        # Create title with additional info if available
        title = f"Drone Defense Simulation - Time: {time:.1f}s"
        if additional_info:
            if "hit_targets" in additional_info:
                title += f" | Hits: {additional_info['hit_targets']}"
            if "expired_targets" in additional_info:
                title += f" | Expired: {additional_info['expired_targets']}"

        self.ax.set_title(title)

        # Plot player position
        self.ax.scatter(
            player_position[0],
            player_position[1],
            player_position[2],
            color="blue",
            marker="o",
            s=200,  # type: ignore
            label="Player",
            edgecolors="black",
            linewidth=2,
        )

        # Plot targets
        for target_id, pos in target_positions.items():
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
            )

        # Add legend if there are targets
        if target_positions or player_position:
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

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
                        pil_frames.append(frame)

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


def get_renderer(renderer: Renderer, *args, **kwargs) -> BaseRenderer:
    """Create a renderer instance."""
    match renderer:
        case "matplotlib":
            cls = MatPlotLibRenderer
        case _:
            raise ValueError(f"Renderer {renderer} is unknown.")
    return cls(*args, **kwargs)
