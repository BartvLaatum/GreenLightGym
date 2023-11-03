import numpy as np

from gymnasium.spaces import Box

class ObservationSpace:
    def __init__(self, model_obs_names, weather_preds) -> None:
        
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(40,))
    
    def _get_observation(self):
        self.GLModel

