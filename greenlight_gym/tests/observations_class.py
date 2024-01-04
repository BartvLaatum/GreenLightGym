from greenlight_gym.envs.greenlight import GreenLightEnv
from greenlight_gym.experiments.utils import load_env_params

env_config_path = f"configs/envs/"
env_id = "GreenLightEnv"
config_name = "four_controls"
env_base_kwargs, env_specific_params, options, state_columns, action_columns = load_env_params(env_id, env_config_path, config_name)

GL = GreenLightEnv(**env_base_kwargs)

obs, info = GL.reset()
try:
    assert (GL._getObs() == obs).all()
except AssertionError:
    print("Assertion error, spaces are not aligned")
    print(GL._getObs(), obs)