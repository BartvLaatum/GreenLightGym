import argparse
from time import time
from typing import List
import numpy as np
import pandas as pd
import sys

from greenlight_gym.experiments.utils import load_env_params, make_env
from greenlight_gym.envs.greenlight import GreenLightEnv
from greenlight_gym.common.results import Results

def run_gl_specified_controls(env: GreenLightEnv, controls: np.ndarray):
    '''
    Runs the GreenLight environment with specified controls.
    Arguments:
        env (GreenLightEnv): The GreenLight environment to run.
        controls (np.ndarray): The control signals to use.
    Returns:
        states (np.ndarray): The states of the environment.
        control_signals (np.ndarray): The control signals used.
        weather (np.ndarray): The weather data.
        timevec (np.ndarray): The time vector.
    '''
    # cast the controls to float32, required by cython model.
    controls = controls.astype(np.float32)
    env.eval_idx = 0
    obs, info = env.reset()
    N = env.N                                       # number of steps
    n_model_vars = env.observations.Nobs            # number of model variables
    states = np.zeros((N+1, n_model_vars))          # array to save states
    control_signals = np.zeros((N, env.GLModel.nu)) # array to save rule-based controls controls

    states[0, :] = obs              # get initial states
    timevec = np.zeros((N+1,))      # array to save time
    timevec[0] = env.GLModel.time
    i=1
    print(f"Running N= {N}, steps")
    while not env.terminated:
        action = controls[i-1]
        obs, r, terminated, _, info = env.step(action)
        states[i, :] += obs[:n_model_vars]
        control_signals[i-1, :] += info["controls"]
        timevec[i] = info["Time"]
        i += 1

    return states[:-1], control_signals, env.weatherData, timevec

def run_store_results(env: GreenLightEnv, controls: np.ndarray, results_columns: List[str]) -> Results:
    results = Results(results_columns)
    states, control_signals, _, _ = run_gl_specified_controls(env, controls)
    data = np.concatenate(([states], [control_signals]), axis=2)
    print(controls.shape)
    print("states", states.shape)
    print(data.shape)
    print(sys.getsizeof(data))
    results.update_result(data)

    return results

    # return states, controls, weather

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightStatesTest")
    parser.add_argument("--step_size", type=str, default="1s")
    parser.add_argument("--date", type=str, default="20000101")
    parser.add_argument("--n_days", type=int, default=10)
    parser.add_argument("--solver", type=str, default="Ode15s")
    parser.add_argument("--order", type=str, default='4th')
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    # configuration file
    env_config_path = "greenlight_gym/configs/envs"
    config_name = "matlab_controls"

    print(f"Running date: {args.date}, and step size: {args.step_size}")

    mat_path = "greenlight_gym/data/model-comparison/matlab"
    py_path = "greenlight_gym/data/model-comparison/python"

    # load in environment parameters
    env_base_params, env_specific_params, options, results_columns = load_env_params(args.env_id, env_config_path, config_name)
    n_days = args.n_days

    # update the step size and the time interval (at which we observe environment variables and send controls)
    env_base_params['h'] = float(args.step_size[:-1])
    env_base_params['time_interval'] = float(args.step_size[:-1])
    env_base_params['season_length'] = n_days

    # load in control and weather data from MATLAB simulation
    controls = pd.read_csv(f"{mat_path}/{args.step_size}StepSizeControls{args.date}{n_days}{args.solver}.csv", sep=",", header=None)
    weather = pd.read_csv(f"{mat_path}/{args.step_size}StepSizeWeather{args.date}{n_days}{args.solver}.csv", sep=",", header=None)

    # add the weather to the environment; such that we don't have to load it in the environment
    env_specific_params['weather'] = weather.values

    env = make_env(args.env_id, rank=0, seed=666, kwargs=env_base_params, kwargsSpecific=env_specific_params, options=options, eval_env=True)()
    results = run_store_results(env, controls.values, results_columns)

    if args.save:
        print("Saving data...")
        results.save(f"{py_path}/{args.step_size}StepSizeResults{args.date}{n_days}{args.solver}{args.order}.csv")