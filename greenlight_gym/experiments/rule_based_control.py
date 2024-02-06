import argparse
from time import time
from typing import List
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from greenlight_gym.experiments.utils import load_env_params, make_env
from greenlight_gym.envs.greenlight import GreenLightEnv

class Results:
    def __init__(self, col_names):
        self.col_names = col_names
        self.results = pd.DataFrame(columns=self.col_names)

    def add_result(self, result):
        self.results = self.results._append(pd.DataFrame(data=result, columns=self.col_names), ignore_index=True)

    def print_head(self, n):
        print(self.results.head(n))

    def save(self, filename):
        self.results.to_csv(filename, index=False)


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
        action = GL.action_space.sample()
        obs, r, terminated, _, info = GL.step(action)
        states[i, :] += obs[:n_model_vars]
        control_signals[i, :] += info["controls"]
        timevec[i] = info["Time"]
        i+=1
    return states, control_signals, GL.weatherData, timevec

def time_experiment(GL, n_runs=1):
    times = np.zeros(n_runs)
    for i, _ in enumerate(range(n_runs)):
        t = time()
        run_rule_based_controller(GL)
        times[i] += time()-t
    return times
    # print(times)
    # print(f"average runtime GL-gym {times.mean()} +- {times.std()}")

def run_store_results(GL, results):
    states, control_signals, weatherData, timevec = run_rule_based_controller(GL)
    # insert time vector into states array
    # states = np.insert(states, timevec, axis=1)

    # compute the difference between the time steps in seconds
    time_diff = np.round(np.diff(timevec)*86400)
    time_diff = np.insert(time_diff, 0, 0)
    cumulative_time = time_diff.cumsum()
    states = np.insert(states, 0, cumulative_time, axis=1)

    results.add_result(states)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightRuleBased")
    parser.add_argument("--HPfolder", type=str, default="GLBase")
    parser.add_argument("--project", type=str, default="GLProduction")
    parser.add_argument("--date", type=str, default="20000101")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--config_name", type=str, default="rule_based_time_exp")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--path", type=str, default="data/model-comparison/python-rule-based/")
    args = parser.parse_args()

    env_config_path = "configs/envs"

    # control frequencies to test in seconds
    # control_frequencies = [2**i for i in range(12)]
    # print(control_frequencies)
    control_frequencies = [2] + [i*30 for i in range(1, 121)]
    time_frequencies = [2, 4, 8, 16] + [i*30 for i in range(1, 121)]

    env_base_params, env_specific_params, options, state_columns, action_columns = load_env_params(args.env_id, env_config_path, args.config_name)
    h = env_base_params["h"]
    control_frequency = env_base_params["time_interval"]
    season_length = env_base_params["season_length"]

    timed_runs = np.zeros(shape=(args.n_runs, len(control_frequencies)))

    for i, control_freq in enumerate(control_frequencies):

        print(f"Running experiment GreenLightGym with control frequency {control_freq} seconds...")
        env_base_params["time_interval"] = control_freq

        GL = make_env(args.env_id, rank=0, seed=666, kwargs=env_base_params, kwargsSpecific=env_specific_params, options=options, eval_env=True)()
        result = Results(["Time [s]"] + GL.observations.model_obs_vars)
        result = run_store_results(GL, results=result)
        filename = f"{h}s-solverstepsize-demands-{args.date}-{season_length}-{control_freq}.csv"

        if args.save:
            print("Saving data...")
            result.save(args.path+filename)

    for i, control_freq in tqdm(enumerate(time_frequencies)):

        print(f"Running experiment GreenLightGym with control frequency {control_freq} seconds...")
        env_base_params["time_interval"] = control_freq

        # for n in range(args.n_runs):
        GL = make_env(args.env_id, rank=0, seed=666, kwargs=env_base_params, kwargsSpecific=env_specific_params, options=options, eval_env=True)()
        timed_runs[:, i] = time_experiment(GL, n_runs=args.n_runs)
    
    timed_results = pd.DataFrame(data=timed_runs, columns=time_frequencies)
    print(timed_results.head(10))

    if args.save:
        timed_results.to_csv(args.path+"2s-timed-runs-1.0.csv", index=False)
    print("Done!")
