import numpy as np
from typing import List, Optional
from gymnasium.spaces import Box, Space
from greenlight_gym.envs.cython import greenlight_cy

class Observations:
    """
    Observer class, which give us control over the observations we want to our RL algorithm to use.
    One can construct observations from GreenLight or from the current and future weather data.
    The model observations are computed in GreenLight,
    the weather observations are extracted from the weather data array.
    
    Args:
        - Nobs (int): number of observations
        - low (float): lower bound for observation space
        - high (float): upper bound for the observation space
        - var_names (List[str]): list of variable names to be observed
    """
    def __init__(self,
                ) -> None:
        self.Nobs = None
        self.low = None
        self.high = None
        self.var_names = None

    def compute_obs(self,
                    GLModel: greenlight_cy.GreenLight,
                    solver_steps: int,
                    weather_data: np.ndarray,
                    ) -> np.ndarray:
        """
        Placeholder function for computing observations.
        """
        pass

class ModelObservations(Observations):
    """
    Model observations module. This class is used to extract observations from the GreenLight model.
    Pass the desired variables from the model to observe in the model_obs_vars list.
    Make sure that the model variables are available as attribute in the Cython code GreenLight model.

    Args:
        - model_obs_vars (List[str]): list of variable names to be observed.
        - low (float): lower bound for observation space
        - high (float): upper bound for the observation space
    """
    def __init__(self,
                model_obs_vars: List[str],      # observations from the model
                low: float=-1e4,                # lower bound for observation space
                high: float=1e4                 # upper bound for the observation space
                ) -> None:
        self.var_names = model_obs_vars
        self.Nobs = len(model_obs_vars)
        self.low = np.full(self.Nobs, low)
        self.high = np.full(self.Nobs, high)

    def compute_obs(self,
                    GLModel: greenlight_cy.GreenLight,
                    solver_steps: int,
                    weather_data: np.ndarray,
                    ) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight model.
        """
        return np.array([getattr(GLModel, model_var) for model_var in self.var_names])

class WeatherObservations(Observations):
    """
    Weather observations module. This class is used to extract observations from the weather data.
    Pass the desired variables from the model to observe in the weather_obs_list list.
    Make sure that the weather variables are available.

    Args:
        - weather_obs_vars (List[str]): list of variable names to be observed.
        - low (float): lower bound for observation space
        - high (float): upper bound for the observation space
    """
    def __init__(self,
                weather_obs_vars: List[str],    # observations from the weather
                Np: int,                        # how many future weather predictions do we observe 
                low: float=-1e4,                # lower bound for observation space
                high: float=1e4                 # upper bound for the observation space
                ) -> None:
        self.var_names = weather_obs_vars
        self.weather_cols = self.weather_vars2idx()
        self.Np = Np
        self.Nobs = len(weather_obs_vars) * (Np + 1)
        self.low = np.full(self.Nobs, low)
        self.high = np.full(self.Nobs, high)

    def weather_vars2idx(self) -> np.ndarray:
        """
        Functions that converts weather variable names to column indices.
        """
        weather_idx = {"glob_rad": 0, "out_temp": 1, "out_rh": 2, "out_co2": 3, "wind_speed": 4,
                       "sky_temp": 5, "out_soil_temp": 6, "dli": 7, "is_day": 8, "is_day_smooth": 9}
        return np.array([weather_idx[weather_var] for weather_var in self.var_names])

    def compute_obs(self,
                    GLModel: greenlight_cy.GreenLight,
                    solver_steps: int,
                    weather_data: np.ndarray,
                    ) -> np.ndarray:
        """
        Retrieve the weather variables.
        """
        weather_idx = np.array([GLModel.timestep*solver_steps] + [(ts + GLModel.timestep)*solver_steps for ts in range(1, self.Np+1)])
        weather_obs = weather_data[weather_idx][:, self.weather_cols].flatten()
        return weather_obs

class StateObservations(Observations):
    """
    State observations module. This class is used to extract the GreenLight model state variables.
    Simply uses all the state variables from the GreenLight model as observation. 
    """
    def __init__(self,
                model_obs_vars: List[str],      # observations from the model
                low: float=-1e4,                # lower bound for observation space
                high: float=1e4                 # upper bound for the observation space
                ) -> None:
        self.var_names = model_obs_vars
        self.Nobs = len(model_obs_vars)
        self.low = np.full(self.Nobs, low)
        self.high = np.full(self.Nobs, high)

    def compute_obs(self,
                    GLModel: greenlight_cy.GreenLight,
                    solver_steps: int,
                    weather_data: np.ndarray,
                    ) -> np.ndarray:
        """
        Retrieve the state variables.
        """
        return GLModel.getStatesArray()

class AggregatedObservations(Observations):
    """
    Aggregated observations module. This class is used to aggregate multiple observation classes.
    This class uses holds a list of observation classes and aggregates them into one observation class.
    Calling the compute_obs method will return the concatenated observations from all the observation classes.
    
    Args:
        - obs_list (List[Observations]): list of observation classes to be aggregated
        - model_obs_idx (int): index of the model observation class in the obs_list
    """
    def __init__(self,
                 obs_list: List[Observations],
                 model_obs_idx: Optional[int] = None) -> None:
        self.obs_list = obs_list
        self.model_obs_idx = model_obs_idx
        self.Nobs = sum([obs.Nobs for obs in obs_list])
        self.low = np.concatenate([obs.low for obs in obs_list])
        self.high = np.concatenate([obs.high for obs in obs_list])

    def compute_obs(self,
                    GLModel: greenlight_cy.GreenLight,
                    solver_steps: int,
                    weather_data: np.ndarray,
                    ) -> np.ndarray:
        """
        Compute, and aggregate observations from GreenLight model and weather.
        """
        return np.concatenate([obs.compute_obs(GLModel, solver_steps, weather_data) for obs in self.obs_list])

if __name__ == "__main__":
    obs_list = [ModelObservations(["air_temp", "air_rh", "co2_ppm", "fruit_weight"]),
                WeatherObservations(["glob_rad", "out_temp", "out_rh", "out_co2", "wind_speed"], Np=1, low=-1e4, high=1e4)]
    obs_class = AggregatedObservations(obs_list)
