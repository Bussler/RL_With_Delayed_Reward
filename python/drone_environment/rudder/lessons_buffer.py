"""Prioritised episode replay buffer for RUDDER.

Stores completed episode trajectories and samples them based on
two priority signals:

1. **Model loss** - episodes the return-predictor finds hard to predict
   are more valuable for training.
2. **Return diversity** - episodes whose return deviates from the buffer
   mean help maintain variance and avoid collapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class EpisodeTrajectory:
    """A complete episode trajectory."""

    observations: list[npt.NDArray[np.float32]]
    actions: list[npt.NDArray[np.float32]]
    rewards: list[float]
    episode_return: float = field(init=False)
    redistribution_loss: float = 1.0  # initial high priority

    def __post_init__(self) -> None:
        """Compute episode return from rewards."""
        self.episode_return = float(np.sum(self.rewards))

    def __len__(self) -> int:
        """Return the number of steps in the episode."""
        return len(self.rewards)


class LessonsBuffer:
    """Fixed-capacity buffer that prioritises informative episodes."""

    def __init__(self, max_size: int = 500) -> None:
        """Initialize the lessons buffer."""
        self.max_size = max_size
        self._buffer: list[EpisodeTrajectory] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, trajectory: EpisodeTrajectory) -> None:
        """Add episode; evict lowest-priority if at capacity."""
        if len(self._buffer) >= self.max_size:
            self._evict_lowest_priority()
        self._buffer.append(trajectory)

    def sample(self, batch_size: int) -> list[EpisodeTrajectory]:
        """Sample episodes with probability proportional to priority."""
        batch_size = min(batch_size, len(self._buffer))
        priorities = np.array([self._priority(ep) for ep in self._buffer], dtype=np.float64)
        priorities /= priorities.sum()
        rng = np.random.default_rng()
        indices = rng.choice(len(self._buffer), size=batch_size, replace=False, p=priorities)
        return [self._buffer[i] for i in indices]

    def update_losses(self, episodes: list[EpisodeTrajectory], losses: list[float]) -> None:
        """Update redistribution loss for the given episodes."""
        for ep, loss in zip(episodes, losses, strict=True):
            ep.redistribution_loss = loss

    @property
    def mean_return(self) -> float:
        """Mean episode return of buffered trajectories."""
        if not self._buffer:
            return 0.0
        return float(np.mean([ep.episode_return for ep in self._buffer]))

    @property
    def return_std(self) -> float:
        """Standard deviation of episode returns in the buffer."""
        if len(self._buffer) < 2:  # noqa: PLR2004
            return 0.0
        return float(np.std([ep.episode_return for ep in self._buffer]))

    def __len__(self) -> int:
        """Return the number of episodes in the buffer."""
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _priority(self, ep: EpisodeTrajectory) -> float:
        """Combined priority: model loss + return diversity."""
        # Diversity: deviation of this episode's return from buffer mean
        mean_ret = self.mean_return
        diversity = abs(ep.episode_return - mean_ret) + 1e-6
        # Model loss scaled so new (high-loss) episodes are preferred
        loss_score = ep.redistribution_loss + 1e-6
        return loss_score + diversity

    def _evict_lowest_priority(self) -> None:
        """Remove the episode with the lowest priority to make room."""
        if not self._buffer:
            return
        worst_idx = int(np.argmin([self._priority(ep) for ep in self._buffer]))
        self._buffer.pop(worst_idx)
