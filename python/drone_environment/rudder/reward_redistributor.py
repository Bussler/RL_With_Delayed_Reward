"""Reward redistribution via RUDDER contribution analysis.

Given a trained :class:`ReturnPredictorLSTM`, this module:

1. Runs the predictor over a trajectory to obtain per-timestep return
   predictions.
2. Computes *consecutive prediction differences* as redistributed rewards
   (contribution analysis).
3. Applies a correction term so the redistributed rewards sum exactly
   to the true episode return.
4. Optionally mixes the redistributed rewards with the original
   environment rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002
import numpy.typing as npt  # noqa: TC002
import torch

if TYPE_CHECKING:
    from drone_environment.rudder.return_predictor import ReturnPredictorLSTM


class RewardRedistributor:
    """Redistribute delayed rewards using RUDDER contribution analysis."""

    def __init__(self, predictor: ReturnPredictorLSTM, alpha: float = 0.3) -> None:
        """Initialize the reward redistributor."""
        self.predictor = predictor
        self.alpha = alpha
        # Online state: previous prediction for step-by-step redistribution
        self._prev_prediction: float = 0.0

    def reset_online_state(self) -> None:
        """Reset the online prediction state at the start of a new episode."""
        self._prev_prediction = 0.0

    @torch.no_grad()
    def redistribute_step(
        self,
        obs: npt.NDArray[np.float32],
        action: npt.NDArray[np.float32],
        device: torch.device | None = None,
    ) -> float:
        """Compute a single-step redistributed reward online.

        Uses the consecutive prediction difference:
        ``r_rudder[t] = pred[t] - pred[t-1]``.

        Call :meth:`reset_online_state` at the start of each episode.

        Args:
            obs:    (obs_dim,) flat observation at current timestep.
            action: (action_dim,) action at current timestep.
            device: torch device.

        Returns:
            Single redistributed reward scalar.
        """
        device = device or next(self.predictor.parameters()).device

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)

        self.predictor.eval()
        pred = self.predictor.predict_sequence(obs_t, act_t).item()  # scalar
        self.predictor.train()

        r_rudder = pred - self._prev_prediction
        self._prev_prediction = pred
        return r_rudder

    @torch.no_grad()
    def redistribute(
        self,
        obs_seq: npt.NDArray[np.float32],
        action_seq: npt.NDArray[np.float32],
        episode_return: float,
        device: torch.device | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute redistributed rewards for one episode.

        Steps:
          1. ``predictions[t]`` = LSTM prediction of episode return at step *t*.
          2. ``r_redist[t] = predictions[t] - predictions[t-1]`` (with ``predictions[-1] = 0``).
          3. Scale ``r_redist`` so ``sum(r_redist) == episode_return``.

        Args:
            obs_seq:       (T, obs_dim) flat observations.
            action_seq:    (T, action_dim) actions (one-hot or raw indices).
            episode_return: true cumulative return of the episode.
            device:        torch device to use for inference.

        Returns:
            redistributed: (T,) numpy array of redistributed rewards.
        """
        device = device or next(self.predictor.parameters()).device

        obs_t = torch.as_tensor(obs_seq, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(action_seq, dtype=torch.float32, device=device)

        self.predictor.eval()
        preds = self.predictor.predict_sequence(obs_t, act_t)  # (T,)
        self.predictor.train()

        # Consecutive differences  (pred[-1] is implicitly 0)
        preds_shifted = torch.cat([torch.zeros(1, device=device), preds[:-1]])
        redistributed = preds - preds_shifted  # (T,)

        # Correction: ensure sum matches true episode return
        raw_sum = redistributed.sum().item()
        if abs(raw_sum) > 1e-8:  # noqa: PLR2004
            correction = episode_return / raw_sum
            redistributed = redistributed * correction
        else:
            # Predictor hasn't learned yet — fall back to uniform spread
            redistributed = torch.full_like(redistributed, episode_return / len(redistributed))

        return redistributed.cpu().numpy()

    def mix_rewards(
        self,
        original: npt.NDArray[np.float64],
        redistributed: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Blend redistributed and original per-step rewards.

        ``mixed = alpha * redistributed + (1 - alpha) * original``
        """
        return self.alpha * redistributed + (1.0 - self.alpha) * original
