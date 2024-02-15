import os
import yaml
import argparse
from copy import copy
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc

import wandb
import numpy as np
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback

from greenlight_gym.common.callbacks import TensorboardCallback
from greenlight_gym.experiments.utils import load_env_params, load_model_params, make_vec_env, set_model_params

#TODO WRITE CUSTOM EXPERIMENT FUNCTION FOR THIS PROCESS.
def run_hp_experiment(
    env_id,
    env_base_params,
    env_specific_params,
    options,
    modelParams,
    SEED,
    n_eval_episodes,
    n_envs,
    project,
    total_timesteps,
    n_evals,
    run=None
    ):

    monitor_filename = None
    vec_norm_kwargs = {"norm_obs": True, "norm_reward": True, "clip_obs": 50_000}


    eval_env = make_vec_env(
        env_id,
        env_base_params,
        env_specific_params,
        options,
        seed=SEED,
        n_envs=10,
        monitor_filename=monitor_filename,
        vec_norm_kwargs=vec_norm_kwargs,
        eval_env=True,
        )

    env_log_dir = f"train_data/{project}/envs/{run.name}/"
    eval_freq = total_timesteps//n_evals//n_envs
    # we don't save the best model here, we just want to evaluate the model
    best_model_save_path = None

    save_name = "vec_norm"
    env = make_vec_env(
            args.env_id,
            env_base_params,
            env_specific_params,
            options,
            seed=SEED,
            n_envs=n_envs,
            monitor_filename=monitor_filename,
            vec_norm_kwargs=vec_norm_kwargs
            )

    callbacks = [TensorboardCallback(
                                eval_env,
                                n_eval_episodes=n_eval_episodes,
                                eval_freq=eval_freq,
                                best_model_save_path=best_model_save_path,
                                name_vec_env=save_name,
                                path_vec_env=env_log_dir,
                                deterministic=True,
                                callback_on_new_best=None,
                                run=None,
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
    
    # run.finish()
    env.close()
    eval_env.close()
    del model
    gc.collect()


def train(config=None):
    with wandb.init(config=config, sync_tensorboard=True) as run:
        config = wandb.config
        modelParams =  set_model_params(config)
        def_config = copy(modelParams)
        def_config['pred_horizon'] = config["pred_horizon"]
        env_base_params["pred_horizon"] = config["pred_horizon"]

        run.config.setdefaults(def_config)    

        run_hp_experiment(
            args.env_id,
            env_base_params,
            env_specific_params,
            options,
            modelParams,
            SEED,
            args.n_eval_episodes,
            config["n_envs"],
            args.project,
            total_timesteps=args.total_timesteps,
            n_evals=args.n_evals,
            run=run,
            )
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightHeatCO2")
    parser.add_argument("--project", type=str, default="tuning-5-min")
    parser.add_argument("--group", type=str, default="testing-evaluation")
    parser.add_argument("--env_config_name", type=str, default="5min_four_controls")
    parser.add_argument("--tuning_file", type=str, default="5-min-four-controls.yml")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_eval_episodes", type=int, default=60)
    parser.add_argument("--n_evals", type=int, default=1)
    parser.add_argument("--method", type=str, default='bayes')
    parser.add_argument("--algorithm", type=str, default="ppo")
    args = parser.parse_args()

    sweep_config = {
        'method': args.method
        }

    metric = {
        'name': 'eval/mean_reward',
        'goal': 'maximize'
        }

    sweep_config['metric'] = metric

    env_config_path = f"configs/envs/"
    model_config_path = f"configs/algorithms/"
    hp_path = f"configs/hp_tuning/"

    SEED = 666
    algorithm = "PPO"
    env_base_params, env_specific_params, options, result_columns = load_env_params(args.env_id, env_config_path, args.env_config_name)
    model_params = load_model_params(args.algorithm, model_config_path, args.env_config_name)

    with open(join(hp_path, args.tuning_file), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    parameters = params["parameters"]

    sweep_config['parameters'] = parameters

    sweep_id = wandb.sweep(sweep_config, project=args.project)
    wandb.agent(sweep_id, train, count=100)
