"""Miscellaneous classes and functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import yaml


def read_yml(f_path: str) -> dict:
    """Read a yaml file to memory."""
    with open(f_path) as f:
        return yaml.safe_load(f)


# Deterministic key order matching DroneGymEnv._create_observation_space()
_OBS_KEY_ORDER: tuple[str, ...] = (
    "player_position",
    "target_positions",
    "target_velocities",
    "target_distances",
    "target_time_remaining",
    "target_death_mask",
    "time_left",
)


def flatten_obs_dict(obs: dict[str, Any]) -> npt.NDArray[np.float32]:
    """Flatten a Dict observation into a 1-D float32 numpy array.

    Keys are iterated in :data:`_OBS_KEY_ORDER` so the layout is
    deterministic and matches the observation space definition.
    """
    parts = [np.asarray(obs[k], dtype=np.float32).ravel() for k in _OBS_KEY_ORDER]
    return np.concatenate(parts)
