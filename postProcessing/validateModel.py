import os
import argparse

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from RLGreenLight.environments.pyutils import days2date
from RLGreenLight.experiments.utils import loadParameters, make_vec_env
from RLGreenLight.experiments.evaluation import evaluate_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="TestVecLoadSave")
    parser.add_argument("--runname", type=str, default="polar-dew-7")
    parser.add_argument("--env_id", type=str, default="GreenLightBase")
    parser.add_argument("--HPfolder", type=str, default="GLBase")
    parser.add_argument("--HPfilename", type=str, default="ppo.yml")
    args = parser.parse_args()

    # hyperparameters
    algorithm = "PPO"
    HPpath = f"hyperparameters/{args.HPfolder}/"
    envParams, envSpecificParams, modelParams, options, state_columns, action_columns =\
                            loadParameters(args.env_id, HPpath, args.HPfilename, algorithm)
    SEED = 666

    vec_norm_kwargs = {"norm_obs": True, "norm_reward": False, "clip_obs": 50_000, "clip_reward": 1000}

    env = make_vec_env(args.env_id, envParams, envSpecificParams, options, seed=SEED, numCpus=1,\
                             monitor_filename=None, vec_norm_kwargs=vec_norm_kwargs, eval_env=True)
    env = VecNormalize.load(f"trainData/{args.project}/envs/{args.runname}/vecnormalize.pkl", env)
    model = PPO.load(f"trainData/{args.project}/models/{args.runname}/best_model.zip", env=env)
    env = model.get_env()

    obs = env.reset()
    obs = obs.reshape(1, -1)

    N = env.get_attr("N")[0]                            # get number of time steps
    states = np.zeros((N+1, envParams["modelObsVars"])) # array to save states
    controlSignals = np.zeros((N, envParams["nu"]))   # array to save rule-based controls controls
    timevec = np.zeros((N+1,))                          # array to save time
    dones = [False]

    states[0, :] = obs[0, :envParams["modelObsVars"]]             # get initial states
    timevec[0] = env.env_method("_getTimeInDays")[0]
    i=0

    episode_rewards, episode_lengths, episode_actions, model_actions, episode_obs, timevec = \
                                                        evaluate_policy(
                                                            model,
                                                            env,
                                                            n_eval_episodes= 1,
                                                            deterministic = True,
                                                            render= False,
                                                            callback = None,
                                                            reward_threshold = None,
                                                            return_episode_rewards = True,
                                                            warn =  True,
                                                            )
    meanActions = np.mean(episode_actions, axis=0)
    meanObs = np.mean(episode_obs, axis=0)

    states = pd.DataFrame(data=meanObs[:-1, :envParams["modelObsVars"]], columns=state_columns)
    controlSignals = pd.DataFrame(data=meanActions, columns=action_columns)

    states["Time"] = np.asarray(days2date(timevec[:-1], "01-01-0001"), "datetime64[s]")
    controlSignals["Time"] = np.asarray(days2date(timevec[:-1], "01-01-0001"), "datetime64[s]")
    dates = states["Time"].dt.strftime("%Y%m%d")

    # check if directory exists
    if not os.path.exists(f"data/ppo/{args.project}/{args.runname}"):
        os.makedirs(f"data/ppo/{args.project}/{args.runname}")


    states.to_csv(f"data/ppo/{args.project}/{args.runname}/states{dates[0]}-{envParams['seasonLength']:03}.csv", index=False)
    controlSignals.to_csv(f"data/ppo/{args.project}/{args.runname}/controls{dates[0]}-{envParams['seasonLength']:03}.csv", index=False)
