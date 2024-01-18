import argparse
from time import time
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from greenlight_gym.experiments.utils import load_env_params, make_env
from greenlight_gym.envs.greenlight import GreenLightEnv

def matlab_controls(GL: GreenLightEnv, controls: List, h_step_sizes: List):
    obs, info = GL.reset()
    N = GL.N                                            # number of steps to take
    n_model_vars = len(GL.observations.model_obs_vars)
    states = np.zeros((N+1, n_model_vars))              # array to save states
    control_signals = np.zeros((N, GL.GLModel.nu))    # array to save rule-based controls controls

    states[0, :] = obs[:n_model_vars]                   # get initial states
    timevec = np.zeros((N+1,))                          # array to save time
    timevec[0] = GL.GLModel.time
    i=1
    while not GL.terminated:
    # for _ in range(2):
        if h_step_sizes.any():
            GL.update_h(h_step_sizes[i-1])

        action = controls[i-1]
        obs, r, terminated, _, info = GL.step(action)
        states[i, :] += obs[:n_model_vars]
        control_signals[i-1, :] += info["controls"]
        timevec[i] = info["Time"]
        i += 1
    return states, control_signals, GL.weatherData, timevec

def run(GL, action_columns, controls, h_step_sizes=None):
    states, control_signals, GL.weatherData, timevec = matlab_controls(GL, controls, h_step_sizes)
    states = np.insert(states, 0, timevec, axis=1)
    states = pd.DataFrame(data=states[:], columns=["Time"] + GL.observations.model_obs_vars)
    controls = pd.DataFrame(data=control_signals, columns=action_columns)
    weather = pd.DataFrame(data=GL.weatherData[[int(ts * GL.GLModel.time_interval/GL.h) for ts in range(0, GL.N)]][:, GL.observations.weather_cols],\
                                columns=GL.observations.weather_obs_vars)
    return states, controls, weather

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightStatesTest")
    parser.add_argument("--step_size", type=str, default="1s")
    parser.add_argument("--date", type=str, default="20000101")
    args = parser.parse_args()

    env_config_path = "configs/envs"
    config_name = "matlab_controls"

    env_base_params, env_specific_params, options, state_columns, action_columns = load_env_params(args.env_id, env_config_path, config_name)
    GL = make_env(args.env_id, rank=0, seed=666, kwargs=env_base_params, kwargsSpecific=env_specific_params, options=options, eval_env=True)()

    if args.step_size == "var":
        h_step_sizes = pd.read_csv(f"data/model_comparison/matlab/stepsize{args.date}.csv", sep=",", header=None).values
        # np.append(h_step_sizes, [0])
        GL.N = len(h_step_sizes)
    # print(h_step_sizes[-1])
    controls = pd.read_csv(f"data/model_comparison/matlab/{args.step_size}StepSizeControls{args.date}.csv", sep=",", header=None)
    weather = pd.read_csv(f"data/model_comparison/matlab/{args.step_size}StepSizeWeather{args.date}.csv", sep=",", header=None)
    states, controls, weather = run(GL, action_columns, controls.values, h_step_sizes=h_step_sizes)

    states.to_csv(f"data/model_comparison/python/{args.step_size}StepSizeStates{args.date}.csv", index=False)
    controls.to_csv(f"data/model_comparison/python/{args.step_size}StepSizeControls{args.date}.csv", index=False)
    weather.to_csv(f"data/model_comparison/python/{args.step_size}StepSizeWeather{args.date}.csv", index=False)
