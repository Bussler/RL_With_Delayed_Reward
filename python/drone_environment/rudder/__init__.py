"""RUDDER: Return Decomposition for Delayed Rewards.

Provides reward redistribution via a learned return-prediction LSTM,
a prioritized lessons buffer for episode trajectories, and a
reward redistributor that mixes RUDDER-attributed rewards with
the original environment rewards.
"""

from drone_environment.rudder.lessons_buffer import EpisodeTrajectory, LessonsBuffer
from drone_environment.rudder.return_predictor import ReturnPredictorLSTM
from drone_environment.rudder.reward_redistributor import RewardRedistributor
from drone_environment.rudder.rudder_wrapper import RudderRewardWrapper

__all__ = [
    "EpisodeTrajectory",
    "LessonsBuffer",
    "ReturnPredictorLSTM",
    "RewardRedistributor",
    "RudderRewardWrapper",
]
