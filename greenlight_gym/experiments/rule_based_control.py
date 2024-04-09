import argparse
from time import time
from typing import Tuple

import numpy as np

from greenlight_gym.experiments.utils import load_env_params, make_env
from greenlight_gym.envs.greenlight import GreenLightEnv
from greenlight_gym.common.results import Results


def run_rule_based_controller(GL: GreenLightEnv) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Run the rule-based controller for the GreenLight environment
    Arguments:
        GL: GreenLightEnv object
    Returns:
        states: np.array of shape (N, n_model_vars) containing the states
        profits: np.array of shape (N, 1) containing the profits
        violations: np.array of shape (N, 3) containing the violations
        timevec: np.array of shape (N, 1) containing the time
    '''
    obs, info = GL.reset()
    N = GL.N                                # number of steps to take
    n_model_vars = GL.observations.Nobs
    states = np.zeros((N+1, n_model_vars))  # array to save states
    control_signals = np.zeros((N+1, GL.GLModel.nu))     # array to save rule-based controls controls

    states[0, :] = obs[:n_model_vars]             # get initial states
    profits = np.zeros((N,1))                   # array to save profit
    violations = np.zeros((N, 3))                # array to save violations
    timevec = np.zeros((N+1,1))                      # array to save time
    timevec[0] = GL.GLModel.time
    i=1

    while not GL.terminated:
        action = GL.action_space.sample()
        obs, r, terminated, _, info = GL.step(action)
        states[i, :] += obs[:n_model_vars]
        control_signals[i, :] += info["controls"]
        profits[i-1] = info["profit"]
        violations[i-1] = info["violations"]
        timevec[i] = info["Time"]
        i+=1
    return states[:-1], profits, violations, timevec[:-1]

def run_store_results(env: GreenLightEnv) -> np.ndarray:
    '''
    Run the rule-based controller for the GreenLight environment and store the results
    Arguments:
        env: GreenLightEnv object
    Returns:
        states: np.array of shape (N, n_model_vars) containing the states
    '''
    t = time()
    states, profits, violations, timevec = run_rule_based_controller(env)
    run_time = time()-t

    # compute the difference between the time steps in seconds
    time_diff = np.round(np.diff(timevec)*86400)
    time_diff = np.insert(time_diff, 0, 0)
    cumulative_time = time_diff.cumsum()
    states = np.insert(states, 0, cumulative_time, axis=1)
    states = np.append(states, profits, axis=1)

    # insert violations (n, 3) into states (n, 8) array
    states = np.concatenate((states, violations), axis=1)
    run_time_vector = np.full((states.shape[0], 1), run_time)
    states = np.append(states, run_time_vector, axis=1)
    return states


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightRuleBased")
    parser.add_argument("--date", type=str, default="20000101")
    parser.add_argument("--config_name", type=str, default="rule_based_time_exp")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--path", type=str, default="data/model-comparison/python-rule-based/")
    args = parser.parse_args()

    env_config_path = "configs/envs"

    control_frequencies = [2, 4, 8, 16] + [i*30 for i in range(1, 121)]
    env_base_params, env_specific_params, options, results_columns = load_env_params(args.env_id, env_config_path, args.config_name)
    result = Results(results_columns)
    control_frequency = env_base_params["time_interval"]
    season_length = env_base_params["season_length"]
    result = Results(results_columns)
    step_size = [0.5, 1.0, 2.0]

    for h in step_size:
        for i, control_freq in enumerate(control_frequencies):

            res = []
            print(f"Running rule-based experiment with control frequency {control_freq}s h {h}s")
            env_base_params["time_interval"] = control_freq
            env_base_params["h"] = h
            GL = make_env(args.env_id, rank=0, seed=666, kwargs=env_base_params, kwargsSpecific=env_specific_params, options=options, eval_env=True)()
            filename = f"step-size-{h}-control-frequency-{control_freq}-{season_length}.csv"

            for start_day in range(len(options['start_days']) - 1):

                print(start_day)
                states = run_store_results(GL)
                # create vector for control_frequency
                control_freq_vector = np.full((states.shape[0], 1), control_freq)
                states = np.append(states, control_freq_vector, axis=1)
                res.append(states)

            result.update_result(np.array(res))

            if args.save:
                print("Saving data...")
                result.save(args.path+filename)
