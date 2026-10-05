"""Versioned tensor contracts shared by training and deployment."""

from .action import ACTION_DIM, CANONICAL_JOINT_ORDER
from .command import COMMAND_DIM
from .observation import ACTOR_OBS_DIM, CRITIC_OBS_DIM

__all__ = [
    "ACTION_DIM",
    "ACTOR_OBS_DIM",
    "CANONICAL_JOINT_ORDER",
    "COMMAND_DIM",
    "CRITIC_OBS_DIM",
]
