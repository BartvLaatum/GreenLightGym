from typing import SupportsFloat, List, Optional
import numpy as np

from greenlight_gym.envs.cython import greenlight_cy

class BaseReward(object):
    """Reward class for the GreenLight environment.
    The reward class is used to compute the reward for the agent based after a single timestep.
    """
    def __init__(self) -> None:
        self.rmin = None
        self.rmax = None
    
    def _scale(self, r: float) -> SupportsFloat:
        return (r - self.rmin)/(self.rmax - self.rmin)

    def _compute_reward(self, GLModel: greenlight_cy.GreenLight) -> SupportsFloat:
        raise NotImplementedError


class HarvestHeatCO2Reward(BaseReward):
    """
    Reward module for the GreenLight heating and carbon dioxide injection environment.
    This module computes the profit-based reward for the RL agent.
    This profit-based reward is computed from revenue from
    harvested tomatoes minus the costs for gas and carbon dioxide.

    Args:
        co2_price (float): price for carbon dioxide [€/kg]
        gas_price (float): price for gas [€/m3]
        tom_price (float): price for tomatoes [€/[FW] kg]
        dmfm (float): dry matter fraction of the fruit [kg [DM]{CH2O} kg^-1 [FW]]
        time_interval (float): time interval for the simulation [h]
        max_co2_rate (float): maximum carbon dioxide rate [kg m^-2 h^-1]
        max_heat_cap (float): maximum heat capacity [J m^-2]
        max_harvest (float): maximum harvest [kg m^-2]
        energy_content_gas (float): energy content of the gas [J m^-3]
    """
    def __init__(self,
                 co2_price: float,
                 gas_price: float,
                 tom_price: float,
                 dmfm: float,
                 time_interval: float,
                 max_co2_rate: float, 
                 max_heat_cap: float,
                 max_harvest: float,
                 energy_content_gas: float,
                 ) -> None:
        self.co2_price = co2_price  # co2 price €/kg
        self.gas_price = gas_price  # gas price €/m3
        self.tom_price = tom_price  # tomato price €/[FW] kg
        self.dmfm = dmfm

        # compute the min and the max reward from profit
        self.rmin = -(max_co2_rate*time_interval*co2_price) -\
            (1e-6*max_heat_cap*time_interval)/energy_content_gas*gas_price   # [€ m^-2]
        self.rmax = max_harvest/self.dmfm * time_interval * tom_price           # [€ m^-2]

    def _compute_reward(self, GLModel: greenlight_cy.GreenLight) -> SupportsFloat:
        """
        Compute the reward for the GreenLight heating and carbon dioxide environment.
        Resource consumption is subtracted from the (cython) GreenLight model object.
        Args:
            GLModel (greenlight_cy.GreenLight): GreenLight model object.

        Returns:
            SupportsFloat: profit-based reward for the agent.
        """
        delta_harvest =  getattr(GLModel, "fruit_harvest")/self.dmfm # [kg [DM]{CH2O} m^-2]
        self.profit = delta_harvest * self.tom_price - getattr(GLModel, "co2_resource")*self.co2_price - getattr(GLModel, "gas_resource") * self.gas_price  # [€ m^-2]
        return self._scale(self.profit)

class ArcTanPenaltyReward(BaseReward):
    """
    Penalty module for the GreenLight environment.
    This module computes the penalty-based reward for the RL agent.
    The penalty-based reward is computed from the absolute difference between the observation and the bounds.
    The penalty is transformed using the inverse tangent function. To bound it between [0, 1].

    Args:
        BaseReward (_type_): _description_
    """
    def __init__(self, 
                 k: List[float],
                 obs_low: List[float],
                 obs_high: List[float]
                 ) -> None:
        self.k = np.array(k)
        self.obs_low = np.array(obs_low)
        self.obs_high = np.array(obs_high)

    def _compute_penalty(self, obs: np.ndarray) -> float:
        """
        Function that computes the penalty for constraint violations.

        Penalty is the absolute difference between the observation and the bounds.
        No penalty is given if the system variables are within the bounds.

        Args:
            obs (np.ndarray): observation of the three indoor variables (temp, co2, rh)
        Returns:
            penalty (np.ndarray): absolute penalty for constraint violations
        """        
        lowerbound = self.obs_low[:] - obs[:]
        lowerbound[lowerbound < 0] = 0
        upperbound = obs[:] - self.obs_high[:]
        upperbound[upperbound < 0] = 0
        return lowerbound + upperbound

    def _compute_reward(self, GLModel: greenlight_cy.GreenLight) -> SupportsFloat:
        """
        Returns the mean of the inverse tangens for absolute penalty values.
        
        Args:
            GLModel (greenlight_cy.GreenLight): GreenLight model object.
        Returns:
            SupportsFloat: penalty-based reward for the agent.
        """ 
        self.abs_pen = self._compute_penalty(GLModel.get_indoor_obs())
        self.pen = 2/np.pi*np.arctan(-self.k*self.abs_pen)
        return np.mean(self.pen)

class AdditiveReward(BaseReward):
    """
    Additive reward module for the GreenLight environment.
    This module computes the additive reward for the RL agent.
    The additive reward is computed from the sum of the rewards from the rewards_list.
    It allows to combine profit-based and penalty-based rewards.

    Args:
        rewards_list (_type_): list of reward modules to be combined
    """
    def __init__(self, rewards_list: List[BaseReward], omega: Optional[float] = None):
        self.rewards_list = rewards_list

    def _compute_reward(self, GLModel: greenlight_cy.GreenLight):
        """
        Compute the addtive reward of the reward modules in the rewards_list.

        Args:
            GLModel (greenlight_cy.GreenLight): GreenLight model object.

        Returns:
            _type_: combined reward for the agent.
        """
        return np.sum([reward._compute_reward(GLModel) for reward in self.rewards_list])

class MultiplicativeReward(BaseReward):
    """
    Multiplicative reward module for the GreenLight environment.
    This module computes the multiplicative reward for the RL agent.
    The multiplicative reward is computed from the product of the rewards from the rewards_list.
    It allows to combine profit-based and penalty-based rewards.
    Combination: profit * (1 - omega * penalty)

    Args:
        reward_list (_type_): list of reward modules to be combined
    """
    def __init__(self, rewards_list: List[BaseReward], omega: float):
        self.rewards_list = rewards_list
        self.omega = omega

    def _compute_reward(self, GLModel: greenlight_cy.GreenLight):
        profit = self.rewards_list[0]._compute_reward(GLModel)
        penalty = self.rewards_list[1]._compute_reward(GLModel)
        return profit * (1.0 - self.omega*(-penalty))

### UNUSED REWARD FUNCTIONS ###
# class LinearScalePenaly(BaseReward):
#     def __init__(self, k: List[float], obs_low: List[float], obs_high: List[float]) -> None:
#         self.k = np.array(k)
#         self.obs_low = np.array(obs_low)
#         self.obs_high = np.array(obs_high)

#     def _compute_penalty(self, obs: np.ndarray) -> float:
#         lowerbound = self.obs_low[:] - obs[:]
#         lowerbound[lowerbound < 0] = 0
#         upperbound = obs[:] - self.obs_high[:]
#         upperbound[upperbound < 0] = 0
#         return lowerbound + upperbound
    
#     def _compute_reward(self, GLModel: greenlight_cy.GreenLight) -> SupportsFloat:
#         self.abs_pen = self._compute_penalty(GLModel.get_indoor_obs())
