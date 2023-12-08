from typing import SupportsFloat

import numpy as np
from gymnasium import RewardWrapper

class BaseReward:
    def __init__(self):
        pass

    def compute_reward(self):
        raise NotImplementedError
    

class CO2Reward(BaseReward):
    def compute_reward(self, obs: np.ndarray)-> SupportsFloat:
        pass

class HeatCO2Reward(BaseReward):
    def __init__(self, co2_price, gas_price, tom_price):
        self.co2_price = co2_price
        self.gas_price = gas_price
        self.tom_price = tom_price

    def _compute_reward(self, obs: np.ndarray) -> SupportsFloat:
        delta_harvest = obs[4]                                                            # [kg [FM] m^-2]
        profit = delta_harvest * self.tom_price - self._get_co2_resource()*self.co2_price - self._get_gas_resource() * self.gas_price  # [€ m^-2]
        return self._scale(self.profit, self.rmin, self.rmax)

class ScaleReward(RewardWrapper):
    """
    A wrapper that min-max scales the reward between [0,1].
    """
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def _compute_reward(self, r: SupportsFloat) -> SupportsFloat:
        return (r - self.min_reward)/(self.max_reward-self.min_reward)
