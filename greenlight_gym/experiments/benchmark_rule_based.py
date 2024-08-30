import os
import argparse

import numpy as np
import pandas as pd

from greenlight_gym.experiments.utils import load_env_params, make_env
from greenlight_gym.envs.greenlight import GreenLightEnv
from greenlight_gym.common.results import Results
from greenlight_gym.common.utils import days2date

def run_rule_based_controller(env: GreenLightEnv):

    obs, info = env.reset()
    print(env.start_day, env.growth_year)

    N = env.N                                # number of steps to take
    observer = env.observations
    n_model_vars = observer.obs_list[0].Nobs
    n_weather_vars = observer.obs_list[1].Nobs
    states = np.zeros((N+1, n_model_vars))  # array to save states
    weather = np.zeros((N+1, n_weather_vars))            # array to save weather
    control_signals = np.zeros((N, env.GLModel.nu))     # array to save rule-based controls controls
    
    episode_profits = np.zeros((N,))  # array to save profits
    episode_violations = np.zeros((N, 3))  # array to save violations

    episode_rewards = np.zeros((N,))  # array to save profits

    states[0, :] = obs[:n_model_vars]             # get initial states
    timevec = np.zeros((N+1,))                      # array to save time
    timevec[0] = env.GLModel.time
    i=0

    while not env.terminated:
        action = env.action_space.sample()
        obs, r, terminated, _, info = env.step(action)
        states[i+1, :] += obs[:n_model_vars]
        weather[i+1] += obs[n_model_vars:]
        control_signals[i, :] += info["controls"]
        timevec[i+1] = info["Time"]
        episode_rewards[i] += r
        episode_profits[i] = info["profit"]
    
        episode_violations[i, :] = info["violations"]
        i+=1

    return states[:-1], control_signals, weather[:-1], timevec[:-1], episode_rewards, episode_profits, episode_violations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightRuleBased")
    parser.add_argument("--n_years", type=int, default=10)
    parser.add_argument("--train_or_test", type=str, default="test")
    parser.add_argument("--config_name", type=str, default="benchmark-rule-based")
    parser.add_argument("--save", action="store_true")
    # parser.add_argument("--results_path", type=str, default="data/{}/rule-based/")
    args = parser.parse_args()
    env_config_path = "greenlight_gym/configs/envs"
    env_base_params, env_specific_params, options, results_columns = load_env_params(args.env_id, env_config_path, args.config_name)

    if args.train_or_test == "train":
        options['growth_years'] = list(range(2011, 2021))

    results_path = f"data/benchmark/{args.train_or_test}/rule-based/"

    h = env_base_params["h"]
    results = Results(results_columns)

    episode_rewards = []
    episode_profits = [] # profits for each episode
    episode_violations = []
    episode_actions = []
    episode_obs = []
    weather_obs = []
    time_vec = []


    envs = [make_env(args.env_id, rank=i, seed=666, kwargs=env_base_params, kwargsSpecific=env_specific_params, options=options, eval_env=True)() for i in range(args.n_years)]

    for i in range(6):
        for env in envs:
            # env._reset_eval_idx()
            states, control_signals, weather, time_vector, rewards, profits, violations = run_rule_based_controller(env)
            
            episode_rewards.append(rewards)
            episode_profits.append(profits)
            episode_violations.append(violations)
            episode_actions.append(control_signals)
            episode_obs.append(states)
            weather_obs.append(weather)
            time_vec.append(time_vector)


    episode_rewards = np.array(episode_rewards)
    episode_profits = np.array(episode_profits)
    episode_violations = np.array(episode_violations)
    episode_actions = np.array(episode_actions)
    episode_obs = np.array(episode_obs)
    weather_obs = np.array(weather_obs)
    time_vec = np.array(time_vec)

    print(episode_rewards.shape)

    episode_returns = np.expand_dims(episode_rewards, axis=-1)
    print(episode_returns.shape)
    times = np.array([pd.to_datetime(days2date(time_vec[i, :], "01-01-0001")).tz_localize("Europe/Amsterdam", nonexistent="shift_forward") for i in range(time_vec.shape[0])])

    # add dimension to time_vec and episode_profits so that they can be concatenated to other arrays
    times = np.expand_dims(times, axis=-1)
    episode_profits = np.expand_dims(episode_profits, axis=-1)
    print(episode_profits.shape)

    # concatenate the results of the current episode to the results of the previous episodes
    data = np.concatenate((times, episode_obs, weather_obs, episode_actions, episode_profits, episode_violations, episode_returns), axis=2)
    results.update_result(data)
    
    if args.save:
        print("saving results")
        # create directory if it does not exist
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        # save results to csv given the run name
        results.save(os.path.join(results_path, f"benchmark-60.csv"))
