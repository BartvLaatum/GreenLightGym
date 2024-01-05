import numpy as np
from typing import List
from gymnasium.spaces import Box, Space
from greenlight_gym.envs.cython import greenlight_cy

class Observations(object):
    """
    Observer class, which give us control over the observations we want to our RL algorithm to use.
    One can construct observations from GreenLight or from the current and future weather data.
    The model observations are computed in GreenLight,
    the weather observations are extracted from the weather data array.
    """
    def __init__(self,
                model_obs_vars: List[str],      # observations from the model
                weather_obs_vars: List[str],    # observations from the weather
                Np: int,                        # how many future weather predictions do we observe 
                low: float=-1e4,                # lower bound for observation space
                high: float=1e4                 # upper bound for the observation space
                ) -> None:
        self.model_obs_vars = model_obs_vars
        self.weather_obs_vars = weather_obs_vars
        self.weather_cols = self.weather_vars2idx()
        self.Np = Np
        self.Nobs = len(model_obs_vars) + len(weather_obs_vars) * (Np + 1)
        self.low = np.full(self.Nobs, low)
        self.high = np.full(self.Nobs, high)

    def weather_vars2idx(self) -> np.ndarray:
        """
        Functions that converts weather variable names to column indices.
        """
        weather_idx = {"glob_rad": 0, "out_temp": 1, "out_rh": 2, "out_co2": 3, "wind_speed": 4,
                       "sky_temp": 5, "out_soil_temp": 6, "dli": 7, "is_day": 8, "is_day_smooth": 9}
        return np.array([weather_idx[weather_var] for weather_var in self.weather_obs_vars])

    def compute_obs(self,
                    GLModel: greenlight_cy.GreenLight,
                    solver_steps: int,
                    weather_data: np.ndarray,
                    ) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        model_obs = [getattr(GLModel, model_var) for model_var in self.model_obs_vars]
        weather_idx = np.array([GLModel.timestep*solver_steps] + [(ts + GLModel.timestep)*solver_steps for ts in range(1, self.Np+1)])
        weather_obs = weather_data[weather_idx][:, self.weather_cols].flatten()
        return np.concatenate([model_obs, weather_obs], axis=0)
