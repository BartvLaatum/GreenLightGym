from typing import Optional, Tuple, Any, List, Dict

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box

from greenlight_gym.envs.cython.greenlight_cy import GreenLight as GL
from greenlight_gym.common.utils import loadWeatherData
from greenlight_gym.envs.observations import ModelObservations, WeatherObservations, AggregatedObservations, StateObservations
from greenlight_gym.envs.rewards import AdditiveReward, HarvestHeatCO2Reward, ArcTanPenaltyReward, MultiplicativeReward

from datetime import date

class GreenLightEnv(gym.Env):
    '''
    Python base class that functions as a wrapper between python and cython. 
    As input we get the weather data, number of variables for states, inputs and disturbances, and whether and which lamps we use.
    Argument definitions are in the init function.
    '''
    def __init__(
                self,
                weather_data_dir:str,       # path to weather data
                location: str,              # location of the recorded weather data
                data_source: str,           # source of the weather data
                h: float,                   # [s] time step for the RK4 solver
                nx: int,                    # number of states 
                nu: int,                    # number of control inputs
                nd: int,                    # number of disturbances
                no_lamps: int,              # whether lamps are used
                led_lamps: int,             # whether led lamps are used
                hps_lamps: int,             # whether hps lamps are used
                int_lamps: int,             # whether interlighting lamps are used
                dmfm: float,                # [kg [FM] m^-2] dry matter fruit mass
                season_length: int,         # [days] length of the growing season
                pred_horizon: int,          # [days] number of future weather predictions
                time_interval: int,         # [s] time interval in between observations
                options: Optional[Dict[str, Any]] = None, # options for the environment (e.g. specify starting date)
                start_train_year: int = 2011,  # start year for training
                end_train_year: int = 2020,    # end year for training
                start_train_day: int = 59,    # end year for training
                end_train_day: int = 244,    # end year for training
                training: bool = True,      # whether we are training or testing
                ) -> None:
        super(GreenLightEnv, self).__init__()

        # number of seconds in the day
        self.c = 86400

        # arguments that are kept the same over various simulations
        self.nx = nx
        self.nu = nu
        self.nd = nd
        self.no_lamps = no_lamps
        self.led_lamps = led_lamps
        self.hps_lamps = hps_lamps
        self.int_lamps = int_lamps
        self.weather_data_dir = weather_data_dir
        self.location = location
        self.data_source = data_source
        self.h = h
        self.dmfm = dmfm
        self.season_length = season_length
        self.pred_horizon = pred_horizon
        self.time_interval = time_interval
        self.N = int(season_length*self.c/time_interval)    # number of timesteps to take for python wrapper
        self.solver_steps = int(time_interval/self.h)       # number of steps the solver takes between time_interval
        # self.options = options
        self.start_train_year = start_train_year
        self.end_train_year = end_train_year
        self.start_train_day = start_train_day
        self.end_train_day = end_train_day
        self.training = training
        self.eval_idx = 0

        self.observations = None
        self.rewards = None

        self.observation_space = None
        self.action_space = None

        # which actuators the GreenLight model can control.        
        self.control_indices = {"uBoil": 0, "uCO2": 1, "uThScr": 2, "uVent": 3, "uLamp": 4, "uIntLamp": 5, "uGroPipe": 6, "uBlScr": 7}

        # lower and upper bounds for air temperature, co2 concentration, humidity
        self.obs_low = None
        self.obs_high = None

        # # initialize the model in cython
        self.GLModel = GL(self.h,
                          nx,
                          nu,
                          self.nd,
                          no_lamps,
                          led_lamps,
                          hps_lamps,
                          int_lamps,
                          self.solver_steps,
                          )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        '''
        Given an action we simulate the next state of the system.
        The system is numerically simulated by the GreenLight model implemented in C.
        We can choose to control the boiler valve, co2, thermal screen, roof vent, lamps, internal lamps, grow pipes and blackout screen.
        '''

        # scale the action to the range of the control inputs, which is between [0, 1]
        action = self._scale(action, self.action_space.low, self.action_space.high)
        self.GLModel.step(action, self.control_idx)
        obs = self._get_obs()
        if self._terminalState(obs):
            self.terminated = True
            reward = 0
        else:
            reward = self._reward()

        # additional information to return
        info = self._get_info()

        return (
                obs,
                reward, 
                self.terminated, 
                False,
                info
                )

    def _get_info(self):
        raise NotImplementedError

    def _init_rewards(self,
                    co2_price: Optional[float] = None,
                    gas_price: Optional[float] = None,
                    tom_price: Optional[float] = None,
                    k: Optional[List[float]] = None,
                    obs_low: Optional[List[float]] = None,
                    obs_high: Optional[List[float]] = None
                    ) -> None:
        raise NotImplementedError

    def _init_observations(self,
                           model_obs_vars: Optional[List[str]] = None,
                           weather_obs_vars: Optional[List[str]] = None,
                           Np: Optional[int] = None
                           ) -> None:
        obs_list = []
        if model_obs_vars is not None:
            obs_list.append(ModelObservations(model_obs_vars))
        if weather_obs_vars is not None:
            obs_list.append(WeatherObservations(weather_obs_vars, Np))
        self.observations = AggregatedObservations(obs_list, model_obs_idx=0)


    def _generate_observation_space(self) -> None:
        self.observation_space = Box(low=self.observations.low,
                                     high=self.observations.high,
                                     shape=(self.observations.Nobs,), 
                                     dtype=np.float32)


    def _get_obs(self):
        raise NotImplementedError

    def _terminalState(self, obs: np.ndarray) -> bool:
        '''
        Function that checks whether the simulation has reached a terminal state.
        Terminal obs are reached when the simulation has reached the end of the growing season.
        Or when there are nan or inf in the state values.
        '''
        if self.GLModel.timestep >= self.N:
            return True
        # check for nan and inf in observation values
        elif np.isnan(obs).any() or np.isinf(obs).any():
            print("Nan or inf in states")
            return True
        return False

    def _get_time(self) -> float:
        return self.GLModel.time

    def _get_time_in_days(self) -> float:
        '''
        Get time in days since 01-01-0001 upto the starting day of the simulation.
        '''
        d0 = date(1, 1, 1)
        d1 = date(self.growth_year, 1, 1)
        delta = d1 - d0
        return delta.days + self.start_day

    def _scale(self, a, amin, amax):
        '''
        Min-max scaler [0,1]. Used for the action space.
        '''
        return (a-amin)/(amax-amin)

    def _reset_eval_idx(self):
        self.eval_idx = 0

    def increase_eval_idx(self):
        self.eval_idx += 1

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        '''
        Container function that resets the environment.
        '''
        super().reset(seed=seed)

        # pick a random growth year and start day if we are training
        if self.training:
            self.growth_year = self.np_random.choice(range(self.start_train_year, self.end_train_year+1))
            self.start_day = self.np_random.choice(range(self.start_train_day, self.end_train_day))          # train 1st January to end of May
        else:
            self.start_day = self.start_days[self.eval_idx]
            self.increase_eval_idx()

        # load in weather data for specific simulation
        self.weatherData = loadWeatherData(
            self.weather_data_dir,
            self.location,
            self.data_source,
            self.growth_year,
            self.start_day,
            self.season_length,
            self.pred_horizon,
            self.h,
            self.nd
            )

        # compute days since 01-01-0001
        # as time indicator by the model
        timeInDays = self._get_time_in_days()

        # reset the GreenLight model starting settings
        self.GLModel.reset(self.weatherData, timeInDays)

        self.terminated = False
        return self._get_obs(), {}

class GreenLightHeatCO2(GreenLightEnv):
    '''
    Child class of GreenLightEnv.
    Starts with a fully mature crop that is ready for harvest.
    The start dates are early year, (January and Februari), which reflects the start of the harvest season.

    Controls the greenhouse climate through four actuators:
    - carbon dioxide supply
    - heating
    - ventialation
    - thermal screen

    Uses a reward that reflects the revenue made from harvesting tomatoes, and the costs for
    heating the greenhouse and injecting CO2 given their resource price.
    Also penalises violating indoor climate boundaries.
    '''
    def __init__(self,
                cLeaf: float = 0.9e5, # [DW] mg/m2
                cStem: float = 2.5e5, # [DW] mg/m2
                cFruit: float = 2.8e5, # [DW] mg/m2
                tCanSum: float = 3e3,
                co2_price: float = 0.1,
                gas_price: float = 0.26,
                tom_price: float = 1.6,
                k: List[float] = [1,1,1],
                obs_low: List[float] = [0, 0, 0],
                obs_high: List[float] = [np.inf, np.inf, np.inf],
                control_signals: Optional[List[str]] = None,
                model_obs_vars: Optional[List[str]] = None,
                weather_obs_vars: Optional[List[str]] = None,
                omega: float = 1.0,
                **kwargs,
                ) -> None:

        super(GreenLightHeatCO2, self).__init__(**kwargs)
        self.cLeaf = cLeaf
        self.cStem = cStem
        self.cFruit = cFruit
        self.tCanSum = tCanSum
        self.co2Price = co2_price
        self.gasPrice = gas_price
        self.tomatoPrice = tom_price
        self.k = np.array(k)
        self.control_signals = control_signals
        self.model_obs_vars = model_obs_vars
        self.weather_obs_vars = weather_obs_vars

        Np = int(self.pred_horizon*self.c/self.time_interval)   # the prediction horizon in timesteps for our weather predictions

        # intialise observation and reward functions
        self._init_observations(model_obs_vars, weather_obs_vars, Np)
        self._init_rewards(co2_price, gas_price, tom_price, k, obs_low, obs_high, omega)

        # initialise the observation and action spaces
        self._generate_observation_space()
        self.action_space = Box(low=-1, high=1, shape=(len(control_signals),), dtype=np.float32)
        self.control_idx = np.array([self.control_indices[control_input] for control_input in control_signals], dtype=np.uint8)

    def _get_info(self):
        return {
            "controls": self.GLModel.getControlsArray(),
            "Time": self.GLModel.time,
            "profit": self.rewards.rewards_list[0].profit,
            "violations": self.rewards.rewards_list[1].abs_pen,
            "timestep": self.GLModel.timestep,
            }

    def _get_obs(self) -> np.ndarray:
        return self.observations.compute_obs(self.GLModel, self.solver_steps, self.weatherData)

    def _init_rewards(self,
                    co2_price: float,
                    gas_price: float,
                    tom_price: float,
                    k: List[float],
                    obs_low: List[float],
                    obs_high: List[float],
                    omega: float = 0.3
                    ) -> None:
        # self.rewards = AdditiveReward([HarvestHeatCO2Reward(co2_price,
        #                                                     gas_price,
        #                                                     tom_price,
        #                                                     self.dmfm,
        #                                                     self.time_interval,
        #                                                     self.GLModel.maxco2rate,
        #                                                     self.GLModel.maxHeatCap,
        #                                                     self.GLModel.maxHarvest,
        #                                                     self.GLModel.energyContentGas),
        #                                 ArcTanPenaltyReward(k,
        #                                                     obs_low, 
        #                                                     obs_high)]
        # )

        self.rewards = MultiplicativeReward([HarvestHeatCO2Reward(
                                                                co2_price,
                                                                gas_price,
                                                                tom_price,
                                                                self.dmfm,
                                                                self.time_interval,
                                                                self.GLModel.maxco2rate,
                                                                self.GLModel.maxHeatCap,
                                                                self.GLModel.maxHarvest,
                                                                self.GLModel.energyContentGas),
                                            ArcTanPenaltyReward(k,
                                                                obs_low, 
                                                                obs_high)],
                                            omega=omega
        )

    def _reward(self) -> float:
        return self.rewards._compute_reward(self.GLModel)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # set crop state to start of the season
        self.GLModel.setCropState(self.cLeaf, self.cStem, self.cFruit, self.tCanSum)
        return self._get_obs(), {}

class GreenLightRuleBased(GreenLightEnv):
    '''
    Child class of GreenLightEnv.
    Starts with a fully mature crop that is ready for harvest.
    The start dates are early year, (January and Februari), which reflects the start of the harvest season.

    Controls the greenhouse climate through four actuators:
    - carbon dioxide supply
    - heating
    - ventialation
    - thermal screen

    Uses a reward that reflects the revenue made from harvesting tomatoes, and the costs for
    heating the greenhouse and injecting CO2 given their resource price.
    Also penalises violating indoor climate boundaries.
    '''
    def __init__(self,
                cLeaf: float = 0.9e5,
                cFruit: float = 2.8e5,
                cStem: float = 2.5e5,
                tCanSum: float  = 1035,
                co2_price: float = 0.1,
                gas_price: float = 0.35,
                tom_price: float = 1.6,
                k: List[float] = [1,1,1],
                obs_low: List[float] = [0, 0, 0],
                obs_high: List[float] = [np.inf, np.inf, np.inf],
                control_signals: List[str] = [],
                model_obs_vars: Optional[List[str]] = None,
                weather_obs_vars: Optional[List[str]] = None,
                **kwargs,
                ) -> None:

        super(GreenLightRuleBased, self).__init__(**kwargs)
        self.cLeaf = cLeaf
        self.cStem = cStem
        self.cFruit = cFruit
        self.tCanSum = tCanSum

        self.control_signals = control_signals
        self.model_obs_vars = model_obs_vars
        self.weather_obs_vars = weather_obs_vars

        Np = int(self.pred_horizon*self.c/self.time_interval)   # the prediction horizon in timesteps for our weather predictions

        # intialise observation and reward functions
        self._init_observations(model_obs_vars, weather_obs_vars, Np)
        self._init_rewards(co2_price, gas_price, tom_price, k, obs_low, obs_high)

        # initialise the observation and action spaces
        self._generate_observation_space()
        self.action_space = Box(low=-1, high=1, shape=(len(control_signals),), dtype=np.float32)
        self.control_idx = np.array([self.control_indices[control_input] for control_input in control_signals], dtype=np.uint8)

    def _get_obs(self) -> np.ndarray:
        return self.observations.compute_obs(self.GLModel, self.solver_steps, self.weatherData)

    def _init_rewards(self,
                    co2_price: float,
                    gas_price: float,
                    tom_price: float,
                    k: List[float],
                    obs_low: List[float],
                    obs_high: List[float]
                    ) -> None:
        self.rewards = AdditiveReward([HarvestHeatCO2Reward(co2_price,
                                                            gas_price,
                                                            tom_price,
                                                            self.dmfm,
                                                            self.time_interval,
                                                            self.GLModel.maxco2rate,
                                                            self.GLModel.maxHeatCap,
                                                            self.GLModel.maxHarvest,
                                                            self.GLModel.energyContentGas),
                                        ArcTanPenaltyReward(k,
                                                            obs_low, 
                                                            obs_high)]
        )

    def _reward(self) -> float:
        return self.rewards._compute_reward(self.GLModel)

    def _get_info(self):
        return {
            "controls": self.GLModel.getControlsArray(),
            "Time": self.GLModel.time,
            "profit": self.rewards.rewards_list[0].profit,
            "violations": self.rewards.rewards_list[1].pen,
            "timestep": self.GLModel.timestep,
            }

    # def _init_observations(self,
    #                        model_obs_vars: List[str],
    #                        weather_obs_vars: List[str],
    #                        Np: int
    #                        ) -> None:
    #     self.observations = ModelObservations(model_obs_vars)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.GLModel.setCropState(self.cLeaf, self.cStem, self.cFruit, self.tCanSum)
        return self._get_obs(), {}

class GreenLightStatesTest(GreenLightEnv):
    '''
    Child class of GreenLightEnv.
    Starts with a fully mature crop that is ready for harvest.
    The start dates are early year, (January and Februari), which reflects the start of the harvest season.

    Controls the greenhouse climate through four actuators:
    - carbon dioxide supply
    - heating
    - ventialation
    - thermal screen

    Uses a reward that reflects the revenue made from harvesting tomatoes, and the costs for
    heating the greenhouse and injecting CO2 given their resource price.
    Also penalises violating indoor climate boundaries.
    '''
    def __init__(self,
                cLeaf: float = 2.5e5,
                cStem: float = 0.9e5,
                cFruit: float = 2.8e5,
                tCanSum: float  = 1035,
                obs_low: List[float] = [0, 0, 0],
                obs_high: List[float] = [np.inf, np.inf, np.inf],
                control_signals: Optional[List[str]] = None,
                model_obs_vars: Optional[List[str]] = None,
                weather_obs_vars: Optional[List[str]] = None,
                weather: np.ndarray = None,
                **kwargs,
                ) -> None:

        super(GreenLightStatesTest, self).__init__(**kwargs)
        self.cLeaf = cLeaf
        self.cStem = cStem
        self.cFruit = cFruit
        self.tCanSum = tCanSum

        self.control_signals = control_signals
        self.model_obs_vars = model_obs_vars
        self.weather_obs_vars = weather_obs_vars

        Np = int(self.pred_horizon*self.c/self.time_interval)   # the prediction horizon in timesteps for our weather predictions

        # intialise observation and reward functions
        self._init_observations(model_obs_vars, weather_obs_vars, Np)

        # initialise the observation and action spaces
        self._generate_observation_space()
        self.action_space = Box(low=0, high=1, shape=(len(control_signals),), dtype=np.float32)
        self.control_idx = np.array([self.control_indices[control_input] for control_input in control_signals], dtype=np.uint8)

        self.weatherData = weather

    def _get_obs(self) -> np.ndarray:
        return self.GLModel.getStatesArray()

    def _reward(self):
        return 1

    def _get_info(self):
        return {
            "controls": self.GLModel.getControlsArray(),
            "Time": self.GLModel.time,
            }

    def update_h(self, h: float):
        self.GLModel.update_h(h)

    def reset(self, 
            seed: Optional[int] = None)\
            -> Tuple[np.ndarray, Dict[str, Any]]:
        '''
        Container function that resets the environment.
        '''
        self.start_day = self.start_days[self.eval_idx]
        self.increase_eval_idx()

        # compute days since 01-01-0001
        # as time indicator by the model
        timeInDays = self._get_time_in_days()
    
        # reset the GreenLight model starting settings
        self.GLModel.reset(self.weatherData, timeInDays)
        self.GLModel.setCropState(self.cLeaf, self.cStem, self.cFruit, self.tCanSum)

        self.terminated = False

        return self._get_obs(), {}

if __name__ == "__main__":
    pass
