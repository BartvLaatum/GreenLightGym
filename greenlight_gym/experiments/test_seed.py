import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from greenlight_gym.envs.greenlight import GreenLightBase, GreenLightCO2, GreenLightHeatCO2
from greenlight_gym.experiments.utils import loadParameters


env_id = "GreenLightHeatCO2"
hpPath = "hyperparameters/gl_heat_co2"
HPfilename = "ppo_4_controls.yml"
algorithm = "PPO"

envBaseParams, envSpecificParams, modelParams, options, state_columns, action_columns =\
                        loadParameters(env_id, hpPath, HPfilename, algorithm)

env = GreenLightHeatCO2(**envBaseParams, **envSpecificParams)
obs, _ =  env.reset(seed=123)
env._reward(obs)

a = np.ones(env.action_space.shape[0], dtype=np.float32)
a[1] = 1
env.step(a)