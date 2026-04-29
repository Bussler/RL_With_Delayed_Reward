"""Strongly-typed configuration models for training scripts.

Usage pattern in a training script:

    cfg = load_config("configs/training/ppo.yaml", overrides, TrainConfig)

Where *overrides* is a list of dot-notation strings such as
``["ppo.learning_rate=1e-3", "experiment.experiment_name=test"]``.
These are collected from ``--set`` CLI flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Logging and checkpointing settings passed to the skrl trainer."""

    experiment_name: str = "experiment"
    directory: str = "skrl/drone_ppo_tensorboard"
    write_interval: int = Field(300, ge=1, description="TensorBoard write interval (seconds).")
    checkpoint_interval: int = Field(10000, ge=1, description="Model checkpoint interval (timesteps).")


class EnvConfig(BaseModel):
    """Drone gymnasium environment settings."""

    drone_config_path: str = "configs/drone_env/default_config.yaml"
    renderer: str = "matplotlib"
    render_mode: str = "rgb_array"

    @field_validator("drone_config_path")
    @classmethod
    def path_must_exist(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"drone_config_path does not exist: {v}")
        return v


class PPOConfig(BaseModel):
    """PPO hyperparameters (mirrors skrl PPO_DEFAULT_CONFIG keys)."""

    rollouts: int = Field(2048, ge=1)
    learning_epochs: int = Field(10, ge=1)
    mini_batches: int = Field(32, ge=1)
    discount_factor: float = Field(0.99, gt=0.0, le=1.0)
    lambda_: float = Field(0.95, gt=0.0, le=1.0, alias="lambda")
    learning_rate: float = Field(3e-4, gt=0.0)
    kl_threshold_scheduler: float = Field(
        0.008, gt=0.0, description="KL threshold for KLAdaptiveRL scheduler."
    )
    grad_norm_clip: float = Field(0.5, ge=0.0)
    ratio_clip: float = Field(0.2, ge=0.0)
    value_clip: float = Field(0.2, ge=0.0)
    clip_predicted_values: bool = True
    entropy_loss_scale: float = Field(0.01, ge=0.0)
    value_loss_scale: float = Field(0.5, ge=0.0)
    use_lstm: bool = True

    model_config = {"populate_by_name": True}


class RudderConfig(BaseModel):
    """RUDDER reward redistribution hyperparameters."""

    alpha: float = Field(
        0.3, ge=0.0, le=1.0, description="Blend weight: alpha*r_rudder + (1-alpha)*r_original."
    )
    warmup_episodes: int = Field(
        50, ge=0, description="Episodes to collect before activating redistribution."
    )
    train_interval: int = Field(10, ge=1, description="Train return predictor every N completed episodes.")
    train_epochs: int = Field(5, ge=1, description="Training passes per predictor training trigger.")
    batch_size: int = Field(16, ge=1, description="Episode batch size sampled from lessons buffer per epoch.")
    buffer_size: int = Field(500, ge=1, description="Maximum episodes stored in the lessons buffer.")
    predictor_lr: float = Field(
        3e-4, gt=0.0, description="Learning rate for the RUDDER return predictor LSTM."
    )
    predictor_hidden: int = Field(128, ge=1, description="LSTM hidden size for the return predictor.")


# ---------------------------------------------------------------------------
# Top-level configs
# ---------------------------------------------------------------------------


class TrainConfig(BaseModel):
    """Top-level training configuration for PPO."""

    model_path: str = Field("", description="Path to pre-trained model. Empty = train from scratch.")
    training_length: int = Field(300000, ge=1)
    eval_length: int = Field(1000, ge=1)
    eval_render_interval: int = Field(5, ge=1)

    experiment: ExperimentConfig = Field()
    env: EnvConfig = Field()
    ppo: PPOConfig = Field()

    model_config = {"populate_by_name": True}


class TrainRudderConfig(TrainConfig):
    """Top-level training configuration for PPO + RUDDER."""

    rudder: RudderConfig = Field()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

# Type alias to keep signatures readable.
type AnyTrainConfig = TrainConfig | TrainRudderConfig


def load_config[T: TrainConfig](
    config_file: str,
    overrides: list[str],
    model: type[T],
) -> T:
    """Load and validate a training config.

    Steps:
      1. Load *config_file* with :func:`OmegaConf.load`.
      2. Merge dot-notation *overrides* (e.g. ``"ppo.learning_rate=1e-3"``).
      3. Convert the merged :class:`DictConfig` to a plain dict.
      4. Validate and return a typed Pydantic model instance.

    Args:
        config_file: Path to a YAML file with default values.
        overrides: List of ``"key.nested=value"`` strings.
        model: Pydantic model class to validate against.

    Returns:
        A validated, strongly-typed configuration object.

    Raises:
        ValidationError: If any value fails Pydantic validation.
        OmegaConfBaseException: If the YAML or an override is malformed.
    """
    base: DictConfig = OmegaConf.load(config_file)  # type: ignore[assignment]
    if overrides:
        patch = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(base, patch)
    else:
        merged = base
    raw: Any = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    return model.model_validate(raw)
