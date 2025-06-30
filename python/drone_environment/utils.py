"""Miscellaneous classes and functions."""

import yaml


def read_yml(f_path: str) -> dict:
    """Read a yaml file to memory."""
    with open(f_path) as f:
        return yaml.safe_load(f)
