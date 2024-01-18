import argparse
from time import time
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from greenlight_gym.experiments.utils import load_env_params, make_env
from greenlight_gym.envs.greenlight import GreenLightEnv

def run_rule_based_controller(GL: GreenLightEnv):
    obs, info = GL.reset()
    N = GL.N                                # number of steps to take
    n_model_vars = len(GL.observations.model_obs_vars)
    states = np.zeros((N+1, n_model_vars))  # array to save states
    control_signals = np.zeros((N+1, GL.GLModel.nu))     # array to save rule-based controls controls

    states[0, :] = obs[:n_model_vars]             # get initial states
    timevec = np.zeros((N+1,))                      # array to save time
    timevec[0] = GL.GLModel.time
    i=1
    while not GL.terminated:
        controls = np.ones((GL.action_space.shape[0],))*0.5
        action = GL.action_space.sample()
        obs, r, terminated, _, info = GL.step(action)
        states[i, :] += obs[:n_model_vars]
        control_signals[i, :] += info["controls"]
        timevec[i] = info["Time"]
        i+=1

    return states, control_signals, GL.weatherData, timevec

def time_experiment(GL, N_runs=1):
    times = np.zeros(N_runs)
    for i, _ in enumerate(range(N_runs)):
        t = time()
        run_rule_based_controller(GL)
        times[i] += time()-t
    print(times)
    print(f"average runtime GL-gym {times.mean()} +- {times.std()}")

def run_save_outputs(GL, action_columns):
    states, control_signals, weatherData, timevec = run_rule_based_controller(GL)
    # insert time vector into states array
    states = np.insert(states, 0, timevec, axis=1)
    states = pd.DataFrame(data=states[:], columns=["Time"]+ GL.observations.model_obs_vars)
    control_signals = pd.DataFrame(data=control_signals, columns=action_columns)
    weather_data = pd.DataFrame(data=weatherData[[int(ts * GL.GLModel.time_interval/GL.h) for ts in range(0, GL.N)]][:, GL.observations.weather_cols],\
                                columns=GL.observations.weather_obs_vars)
    # save outputs
    states.to_csv("data/model_comparison/python/1sStepSizeStates.csv", index=False)
    control_signals.to_csv("data/model_comparison/python/1sStepSizeControls.csv", index=False)
    weather_data.to_csv("data/model_comparison/python/1sStepSizeWeather.csv", index=False)
    # # return states, control_signals, weather_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightRuleBased")
    parser.add_argument("--HPfolder", type=str, default="GLBase")
    parser.add_argument("--project", type=str, default="GLProduction")
    args = parser.parse_args()

    env_config_path = "configs/envs"
    config_name = "rule_based"

    env_base_params, env_specific_params, options, state_columns, action_columns = load_env_params(args.env_id, env_config_path, config_name)
    GL = make_env(args.env_id, rank=0, seed=666, kwargs=env_base_params, kwargsSpecific=env_specific_params, options=options, eval_env=True)()
    N_runs = 10
    # time_experiment(GL, N_runs)
    run_save_outputs(GL, action_columns)
