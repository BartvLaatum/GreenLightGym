import os
import argparse

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from greenlight_gym.common.utils import days2date
from greenlight_gym.common.results import Results
from greenlight_gym.experiments.utils import load_env_params, load_model_params, make_vec_env
from greenlight_gym.common.evaluation import evaluate_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="benchmark-ppo")
    parser.add_argument("--group", type=str, help="name of the experiment group of models you wish to validate.")
    parser.add_argument("--runname", type=str, default="winter-capibara-99")
    parser.add_argument("--validation_type", type=str, help="type of validation to perform either 'train' or 'test'")
    parser.add_argument("--best_or_last", type=str, help="type of model to validate either 'best' or 'last'")
    parser.add_argument("--n_eval_episodes", type=int, default=60)
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--env_id", type=str, default="GreenLightEnv")
    parser.add_argument("--config_filename", type=str, default="multiplicative_pen_daily_avg_temp")
    args = parser.parse_args()

    # hyperparameters
    algorithm = "PPO"
    hp_path = f"configs/envs/"
    results_path = f"data/{args.project}/{args.validation_type}/{args.group}/"
    SEED = 666
    env_base_params, env_specific_params, options, results_columns = load_env_params(args.env_id, hp_path, args.config_filename)

    # options['start_days'] = [59, 74, 90, 105, 120, 135, 151, 166, 181, 196, 212, 237, 243]
    if args.validation_type == "train":
        options['growth_years'] = list(range(2011, 2021))

    if args.n_eval_episodes == 120:
        options['start_days'] =  [59, 74, 90, 105, 120, 135, 151, 166, 181, 196, 212, 237, 243]

    vec_norm_kwargs = {"norm_obs": True, "norm_reward": False, "clip_obs": 50_000, "clip_reward": 1000}

    print(os.path.join(results_path, f"{args.runname}-{args.n_eval_episodes}.csv"))

    env = make_vec_env(args.env_id, env_base_params, env_specific_params, options, seed=SEED, n_envs=10,\
                             monitor_filename=None, vec_norm_kwargs=vec_norm_kwargs, eval_env=True)
    model_name = f"{args.best_or_last}_model.zip"

    if args.best_or_last == "best":
        vec_name = f"vecnormalize.pkl"

    elif args.best_or_last == "last":
        vec_name = f"{args.best_or_last}_vecnormalize.pkl"

    env = VecNormalize.load(f"train_data/{args.project}/envs/{args.runname}/{vec_name}", env)
    model = PPO.load(f"train_data/{args.project}/models/{args.runname}/{model_name}", env=env)

    env.env_method("_reset_eval_idx")
    episode_rewards, episode_std_rewards, episode_actions, episode_obs, time_vec, episode_profits, episode_violations = \
                                                        evaluate_policy(
                                                            model,
                                                            env,
                                                            n_eval_episodes= args.n_eval_episodes,
                                                            deterministic = True,
                                                            render= False,
                                                            callback = None,
                                                            reward_threshold = None,
                                                            return_episode_rewards = True,
                                                            warn =  True,
                                                            )

    # results_columns += ['Final return']
    results = Results(results_columns)

    episode_returns = np.expand_dims(np.tile(episode_rewards, (episode_obs.shape[1], 1)).T, axis=-1)
    times = np.array([pd.to_datetime(days2date(time_vec[i, :], "01-01-0001")).tz_localize("Europe/Amsterdam", nonexistent='shift_forward') for i in range(time_vec.shape[0])])

    # add dimension to time_vec and episode_profits so that they can be concatenated to other arrays
    times = np.expand_dims(times, axis=-1)
    episode_profits = np.expand_dims(episode_profits, axis=-1)

    # concatenate the results of the current episode to the results of the previous episodes
    data = np.concatenate((times,  episode_obs[:,:,:], episode_actions, episode_profits, episode_violations, episode_returns), axis=2)
    results.update_result(data)
    if args.save_results:
        print('saving results')
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        # save results to csv given the run name
        results.save(os.path.join(results_path, f"{args.runname}-{args.n_eval_episodes}-{args.best_or_last}.csv"))
