"""LSTM-based return predictor for RUDDER reward redistribution.

The model processes (observation, action) sequences and predicts the
episode return.  Two losses are combined:

* **Main task** - MSE of the final-timestep prediction vs. actual return.
* **Auxiliary task** - MSE of *every*-timestep prediction vs. actual return,
  down-weighted by ``auxiliary_loss_weight``.  This encourages the model to
  push return-predictive information earlier in the sequence, enabling
  finer-grained contribution analysis.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class ReturnPredictorLSTM(nn.Module):
    """Sequence model that predicts episode return from (obs, action) trajectories."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        encoder_hidden_size: int = 64,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        learning_rate: float = 1e-3,
        auxiliary_loss_weight: float = 0.1,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the return predictor LSTM."""
        super().__init__()

        input_dim = obs_dim + action_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(encoder_hidden_size, lstm_hidden_size),
            nn.LeakyReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(lstm_hidden_size, 1)

        self.auxiliary_loss_weight = auxiliary_loss_weight

        # Move to device *before* creating the optimizer so that
        # parameter references are on the correct device.
        if device is not None:
            self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict_sequence(
        self,
        obs_seq: Tensor,
        action_seq: Tensor,
    ) -> Tensor:
        """Return per-timestep return predictions for a *single* episode.

        Args:
            obs_seq:    (T, obs_dim)
            action_seq: (T, action_dim)

        Returns:
            predictions: (T,) - predicted episode return at each timestep.
        """
        x = torch.cat([obs_seq, action_seq], dim=-1)  # (T, input_dim)
        x = self.encoder(x).unsqueeze(0)  # (1, T, H)
        lstm_out, _ = self.lstm(x)  # (1, T, H)
        return self.head(lstm_out).squeeze(0).squeeze(-1)  # (T,)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_batch(
        self,
        obs_seqs: list[Tensor],
        action_seqs: list[Tensor],
        returns: Tensor,
        lengths: list[int],
    ) -> float:
        """Train on a batch of variable-length episodes.

        Args:
            obs_seqs:    list of (T_i, obs_dim) tensors.
            action_seqs: list of (T_i, action_dim) tensors.
            returns:     (B,) actual episode returns.
            lengths:     list of episode lengths.

        Returns:
            Scalar training loss.
        """
        device = returns.device

        # Concatenate obs + action, then pad for batching
        input_seqs = [torch.cat([o, a], dim=-1) for o, a in zip(obs_seqs, action_seqs, strict=True)]
        padded = pad_sequence(input_seqs, batch_first=True)  # (B, T_max, D)
        encoded = self.encoder(padded)  # (B, T_max, H)

        length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
        packed = pack_padded_sequence(
            encoded,
            length_tensor.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        lstm_out_packed, _ = self.lstm(packed)

        # Unpack to padded tensor
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)  # (B, T_max, H)

        all_preds = self.head(lstm_out).squeeze(-1)  # (B, T_max)

        # --- Main loss: final-timestep prediction vs. actual return ---
        batch_idx = torch.arange(len(lengths), device=device)
        final_idx = length_tensor - 1
        final_preds = all_preds[batch_idx, final_idx]  # (B,)
        main_loss = nn.functional.mse_loss(final_preds, returns)

        # --- Auxiliary loss: every timestep prediction vs. actual return ---
        #     Masked so padding positions don't contribute.
        max_len = all_preds.shape[1]
        time_idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T_max)
        mask = time_idx < length_tensor.unsqueeze(1)  # (B, T_max)

        targets = returns.unsqueeze(1).expand_as(all_preds)  # (B, T_max)
        aux_loss = (nn.functional.mse_loss(all_preds, targets, reduction="none") * mask).sum()
        aux_loss = aux_loss / mask.sum()

        loss = main_loss + self.auxiliary_loss_weight * aux_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
