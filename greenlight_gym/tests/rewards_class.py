import numpy as np

from greenlight_gym.envs.greenlight import GreenLightHeatCO2
from greenlight_gym.experiments.utils import load_env_params

env_config_path = f"configs/envs/"
env_id = "GreenLightHeatCO2"
config_name = "four_controls"
env_base_kwargs, env_specific_params, options, state_columns, action_columns = load_env_params(env_id, env_config_path, config_name)

GL = GreenLightHeatCO2(**env_specific_params,**env_base_kwargs)

obs, info = GL.reset()

for _ in range(92):
    obs, r, done, info, truncated = GL.step(GL.action_space.sample())
