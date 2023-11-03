from typing import SupportsFloat

import numpy as np
from gymnasium import RewardWrapper



class BaseReward:
    def get_reward_function(self):
        raise NotImplementedError
    

class CO2Reward(BaseReward):
    def _reward(self, obs):
        super().__init__(env)

class ScaleReward(RewardWrapper):
    """
    A wrapper that scales the reward between 0,1
    """
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, r: SupportsFloat) -> SupportsFloat:
        return (r - self.min_reward)/(self.max_reward-self.min_reward)
