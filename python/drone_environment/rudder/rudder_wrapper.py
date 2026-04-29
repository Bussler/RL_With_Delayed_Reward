"""Gymnasium wrapper that applies RUDDER reward redistribution.

Wraps any :class:`gymnasium.Env` with Dict observations and continuous
actions.  During training the wrapper:

1. Collects full episode trajectories into a :class:`LessonsBuffer`.
2. Periodically trains a :class:`ReturnPredictorLSTM` on buffered episodes.
3. Uses :class:`RewardRedistributor` to replace the per-step reward with
   a RUDDER-attributed signal (consecutive LSTM prediction differences),
   mixed with the original reward via blending parameter *alpha*.

The wrapper is transparent to the RL algorithm — it only modifies the
scalar reward returned by ``step()``.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch

from drone_environment.rudder.lessons_buffer import EpisodeTrajectory, LessonsBuffer
from drone_environment.rudder.return_predictor import ReturnPredictorLSTM
from drone_environment.rudder.reward_redistributor import RewardRedistributor
from drone_environment.utils import flatten_obs_dict

logger = logging.getLogger(__name__)


class RudderRewardWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """Gymnasium wrapper that redistributes rewards using RUDDER.

    Args:
        env: The environment to wrap.
        obs_dim: Dimensionality of flattened observations.
        action_dim: Dimensionality of the action space.
        alpha: Blending weight — ``alpha * r_rudder + (1 - alpha) * r_original``.
        warmup_episodes: Number of episodes to collect before activating redistribution.
        train_interval: Train the return predictor every *N* completed episodes.
        train_epochs: Number of training passes per training trigger.
        batch_size: Episode batch size sampled from the lessons buffer per epoch.
        buffer_size: Maximum number of episodes stored in the lessons buffer.
        predictor_lr: Learning rate for the return predictor LSTM.
        predictor_hidden: LSTM hidden size for the return predictor.
        auxiliary_loss_weight: Weight for the auxiliary (every-timestep) prediction loss.
        device: Torch device for the predictor.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_dim: int,
        action_dim: int,
        *,
        alpha: float = 0.3,
        warmup_episodes: int = 50,
        train_interval: int = 10,
        train_epochs: int = 5,
        batch_size: int = 16,
        buffer_size: int = 500,
        predictor_lr: float = 1e-3,
        predictor_hidden: int = 128,
        auxiliary_loss_weight: float = 0.1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(env)

        self.device = device or torch.device("cpu")
        self.alpha = alpha
        self.warmup_episodes = warmup_episodes
        self.train_interval = train_interval
        self.train_epochs = train_epochs
        self.batch_size = batch_size

        # RUDDER components
        self.predictor = ReturnPredictorLSTM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lstm_hidden_size=predictor_hidden,
            learning_rate=predictor_lr,
            auxiliary_loss_weight=auxiliary_loss_weight,
            device=self.device,
        )

        self.buffer = LessonsBuffer(max_size=buffer_size)
        self.redistributor = RewardRedistributor(self.predictor, alpha=alpha)

        # Episode accumulators
        self._ep_obs: list[npt.NDArray[np.float32]] = []
        self._ep_actions: list[npt.NDArray[np.float32]] = []
        self._ep_rewards: list[float] = []

        # Counters
        self._completed_episodes: int = 0
        self._total_predictor_loss: float = 0.0
        self._predictor_train_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and clear episode buffers."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._ep_obs.clear()
        self._ep_actions.clear()
        self._ep_rewards.clear()
        self.redistributor.reset_online_state()
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Take a step, optionally replacing the reward with RUDDER attribution."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        original_reward = float(reward)

        # Flatten observation and action for RUDDER storage
        flat_obs = flatten_obs_dict(obs)
        flat_action = np.asarray(action, dtype=np.float32).ravel()

        # Store transition (always using the *original* reward)
        self._ep_obs.append(flat_obs)
        self._ep_actions.append(flat_action)
        self._ep_rewards.append(original_reward)

        # Compute redistributed reward if past warmup
        rudder_active = self._completed_episodes >= self.warmup_episodes
        if rudder_active:
            r_rudder = self.redistributor.redistribute_step(flat_obs, flat_action, device=self.device)
            reward = self.alpha * r_rudder + (1.0 - self.alpha) * original_reward
        else:
            reward = original_reward

        # On episode end: store trajectory and maybe train predictor
        done = terminated or truncated
        if done:
            self._on_episode_end()

        info["rudder_original_reward"] = original_reward
        info["rudder_active"] = rudder_active

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_episode_end(self) -> None:
        """Store the completed episode and optionally train the predictor."""
        if len(self._ep_rewards) == 0:
            return

        trajectory = EpisodeTrajectory(
            observations=list(self._ep_obs),
            actions=list(self._ep_actions),
            rewards=list(self._ep_rewards),
        )
        self.buffer.add(trajectory)
        self._completed_episodes += 1

        # Train predictor periodically
        if (
            self._completed_episodes >= self.warmup_episodes
            and self._completed_episodes % self.train_interval == 0
            and len(self.buffer) >= self.batch_size
        ):
            self._train_predictor()

    def _train_predictor(self) -> None:
        """Sample from buffer and train the return predictor.

        Note: skrl calls ``env.step()`` inside ``torch.no_grad()``, so we must
        explicitly re-enable gradients here for the predictor backward pass.
        """
        with torch.enable_grad():
            for _ in range(self.train_epochs):
                episodes = self.buffer.sample(self.batch_size)

                obs_seqs = [
                    torch.as_tensor(np.stack(ep.observations), dtype=torch.float32, device=self.device)
                    for ep in episodes
                ]
                action_seqs = [
                    torch.as_tensor(np.stack(ep.actions), dtype=torch.float32, device=self.device)
                    for ep in episodes
                ]
                returns = torch.tensor(
                    [ep.episode_return for ep in episodes],
                    dtype=torch.float32,
                    device=self.device,
                )
                lengths = [len(ep) for ep in episodes]

                loss = self.predictor.train_on_batch(obs_seqs, action_seqs, returns, lengths)
                self._total_predictor_loss += loss
                self._predictor_train_count += 1

                # Update per-episode losses in buffer for prioritisation
                per_ep_losses = self._compute_per_episode_losses(episodes, obs_seqs, action_seqs, returns)
                self.buffer.update_losses(episodes, per_ep_losses)

        avg_loss = self._total_predictor_loss / max(self._predictor_train_count, 1)
        logger.info(
            "RUDDER predictor trained | episodes=%d | buffer=%d | avg_loss=%.4f",
            self._completed_episodes,
            len(self.buffer),
            avg_loss,
        )

    @torch.no_grad()
    def _compute_per_episode_losses(
        self,
        episodes: list[EpisodeTrajectory],
        obs_seqs: list[torch.Tensor],
        action_seqs: list[torch.Tensor],
        returns: torch.Tensor,
    ) -> list[float]:
        """Compute individual episode prediction losses for buffer prioritisation."""
        self.predictor.eval()
        losses: list[float] = []
        for i, ep in enumerate(episodes):
            preds = self.predictor.predict_sequence(obs_seqs[i], action_seqs[i])
            final_pred = preds[-1]
            loss = (final_pred - returns[i]).pow(2).item()
            losses.append(loss)
        self.predictor.train()
        return losses
