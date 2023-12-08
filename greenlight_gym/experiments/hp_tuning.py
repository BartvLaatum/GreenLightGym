import os
import yaml
import argparse
from copy import copy
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import wandb
import numpy as np
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback

from greenlight_gym.common.callbacks import TensorboardCallback
from greenlight_gym.experiments.utils import loadParameters, make_vec_env, set_model_params

#TODO WRITE CUSTOM EXPERIMENT FUNCTION FOR THIS PROCESS.
def run_hp_experiment(
    env_id,
    envBaseParams,
    envSpecificParams,
    options,
    modelParams,
    SEED,
    n_eval_episodes,
    n_envs,
    project,
    total_timesteps,
    n_evals,
    state_columns,
    action_columns,
    states2plot=None,
    actions2plot=None,
    run=None
    ):

    monitor_filename = None
    vec_norm_kwargs = {"norm_obs": True, "norm_reward": True, "clip_obs": 50_000}

    eval_env = make_vec_env(
        env_id,
        envBaseParams,
        envSpecificParams,
        options,
        seed=SEED,
        numCpus=1,
        monitor_filename=monitor_filename,
        vec_norm_kwargs=vec_norm_kwargs,
        eval_env=True,
        )

    env_log_dir = f"train_data/{project}/envs/{run.name}/"
    model_log_dir = f"train_data/{project}/models/{run.name}/"
    eval_freq = total_timesteps//n_evals//n_envs

    save_name = "vec_norm"
    env = make_vec_env(
            args.env_id,
            envBaseParams,
            envSpecificParams,
            options,
            seed=SEED,
            numCpus=n_envs,
            monitor_filename=monitor_filename,
            vec_norm_kwargs=vec_norm_kwargs
            )

    callbacks = [TensorboardCallback(
                                eval_env,
                                n_eval_episodes=n_eval_episodes,
                                eval_freq=eval_freq,
                                best_model_save_path=model_log_dir,
                                name_vec_env=save_name,
                                path_vec_env=env_log_dir,
                                deterministic=True,
                                callback_on_new_best=None,
                                run=None,
                                action_columns=action_columns,
                                state_columns=state_columns,
                                states2plot=None,
                                actions2plot=None,
                                verbose=1),
                                WandbCallback(verbose=1)
                                ]

    tensorboard_log = f"train_data/{project}/logs/{run.name}"

    model = PPO(
        env=env,
        seed=SEED,
        verbose=0,
        **modelParams,
        tensorboard_log=tensorboard_log
        )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
        )

def train(config=None):
    with wandb.init(config=config, sync_tensorboard=True) as run:
        config = wandb.config
        modelParams =  set_model_params(config)
        def_config = copy(modelParams)
        def_config['predHorizon'] = config["predHorizon"]
        envBaseParams["predHorizon"] = config["predHorizon"]

        run.config.setdefaults(def_config)    

        run_hp_experiment(
            args.env_id,
            envBaseParams,
            envSpecificParams,
            options,
            modelParams,
            SEED,
            args.n_eval_episodes,
            config["n_envs"],
            args.project,
            total_timesteps=args.total_timesteps,
            n_evals=args.n_evals,
            state_columns=state_columns,
            action_columns=action_columns,
            states2plot=None,
            actions2plot=None,
            run=run,
            )
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightHeatCO2")
    parser.add_argument("--project", type=str, default="TestVecLoadSave")
    parser.add_argument("--group", type=str, default="testing-evaluation")
    parser.add_argument("--HPfolder", type=str, default="gl_heat_co2")
    parser.add_argument("--HPfilename", type=str, default="ppo_4_controls.yml")
    parser.add_argument("--tuning_file", type=str, default="tuning.yml")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--n_evals", type=int, default=10)
    args = parser.parse_args()

    sweep_config = {
        'method': 'bayes'
        }

    metric = {
        'name': 'rollout/ep_rew_mean',
        'goal': 'maximize'
        }

    early_terminate = {
        "type": "hyperband",
        "min_iter": 50,
        "eta": 3,
        }
    sweep_config['metric'] = metric
    sweep_config["early_terminate"] = early_terminate

    hpPath = f"hyperparameters/{args.HPfolder}/"
    states2plot = ["Air Temperature","CO2 concentration", "Humidity", "Fruit harvest", "PAR", "Cumulative harvest"]
    actions2plot = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp"]

    SEED = 666
    algorithm = "PPO"
    envBaseParams, envSpecificParams, modelParams, options, state_columns, action_columns =\
                            loadParameters(args.env_id, hpPath, args.HPfilename, algorithm)

    with open(join(hpPath, args.tuning_file), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    parameters = params["parameters"]

    sweep_config['parameters'] = parameters

    sweep_id = wandb.sweep(sweep_config, project=args.project)
    wandb.agent(sweep_id, train, count=100)
