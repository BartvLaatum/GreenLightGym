import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from pprint import pprint

import wandb
import numpy as np
from stable_baselines3 import PPO

from RLGreenLight.callbacks.customCallback import TensorboardCallback
from RLGreenLight.experiments.utils import loadParameters, make_vec_env, create_callbacks, set_model_params, make_env

os.environ["WANDB_DEBUG"] = "true"

#TODO WRITE CUSTOM EXPERIMENT FUNCTION FOR THIS PROCESS.
def run_hp_experiment(
    env_id,
    envBaseParams,
    envSpecificParams,
    options,
    modelParams,
    SEED,
    n_eval_episodes,
    numCpus,
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
        numCpus=2,
        monitor_filename=monitor_filename,
        vec_norm_kwargs=vec_norm_kwargs,
        eval_env=True,
        )

    env_log_dir = f"trainData/{project}/envs/{run.name}/"
    model_log_dir = f"trainData/{project}/models/{run.name}/"
    eval_freq = total_timesteps//n_evals//numCpus
    save_name = "vec_norm"
    env = make_vec_env(
            args.env_id,
            envBaseParams,
            envSpecificParams,
            options,
            seed=SEED,
            numCpus=args.numCpus,
            monitor_filename=monitor_filename,
            vec_norm_kwargs=vec_norm_kwargs
            )

    # callbacks = TensorboardCallback(eval_env,
    #                                     n_eval_episodes=n_eval_episodes,
    #                                     eval_freq=eval_freq,
    #                                     best_model_save_path=model_log_dir,
    #                                     name_vec_env=save_name,
    #                                     path_vec_env=env_log_dir,
    #                                     deterministic=True,
    #                                     callback_on_new_best=None,
    #                                     run=None,
    #                                     action_columns=action_columns,
    #                                     state_columns=state_columns,
    #                                     states2plot=None,
    #                                     actions2plot=None,
    #                                     verbose=1)
                                        

    tensorboard_log = f"trainData/{project}/logs/{run.name}"
    print(modelParams)
    model = PPO(
        env=env,
        seed=SEED,
        verbose=1,
        **modelParams,
        tensorboard_log=tensorboard_log
        )
    print(model)

    print("whut")
    model.learn(total_timesteps=1)
    print("erreur")
    return
    # model.learn(total_timesteps=args.total_timesteps, 
    #             callback=[])
    # run.fin

def train(config=None):
    SEED = 666
    with wandb.init(config=config):
        config = wandb.config
        config =  set_model_params(config)
        monitor_filename = None
        vec_norm_kwargs = {"norm_obs": True, "norm_reward": True, "clip_obs": 50_000}

        # env = make_vec_env(
        #     args.env_id,
        #     envBaseParams,
        #     envSpecificParams,
        #     options,
        #     seed=SEED,
        #     numCpus=args.numCpus,
        #     monitor_filename=monitor_filename,
        #     vec_norm_kwargs=vec_norm_kwargs
        #     )
        env_fn = make_env(
            args.env_id,
            SEED,
            SEED,
            envBaseParams,
            envSpecificParams,
            options,
            eval_env=False)
        model = PPO(
        env=env_fn(),
        seed=SEED,
        verbose=1,
        **config,
        )
        model.learn(total_timesteps=1)

        wandb.log({"eval/mean_reward": 10})

        # run_hp_experiment(
        #     args.env_id,
        #     envBaseParams,
        #     envSpecificParams,
        #     options,
        #     config,
        #     SEED,
        #     args.n_eval_episodes,
        #     args.numCpus,
        #     args.project,
        #     total_timesteps=10_000,
        #     n_evals=1,
        #     state_columns=state_columns,
        #     action_columns=action_columns,
        #     states2plot=None,
        #     actions2plot=None,
        #     run=wandb.run,
        #     )
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightBase")
    parser.add_argument("--project", type=str, default="TestVecLoadSave")
    parser.add_argument("--group", type=str, default="testing-evaluation")
    parser.add_argument("--HPfolder", type=str, default="GLBase/ppo")
    parser.add_argument("--HPfilename", type=str, default="ppo.yml")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--numCpus", type=int, default=4)
    parser.add_argument("--n_evals", type=int, default=10)
    args = parser.parse_args()

    sweep_config = {
        'method': 'random'
        }

    # metric = {
    #     'name': 'eval/mean_reward',
    #     'goal': 'maximize'   
    #     }

    # sweep_config['metric'] = metric

    parameter_dict = {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,
            }
    }

    hpPath = f"hyperparameters/{args.HPfolder}/"
    states2plot = ["Air Temperature","CO2 concentration", "Humidity", "Fruit harvest", "PAR", "Cumulative harvest"]
    actions2plot = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp"]

    SEED = 666
    algorithm = "PPO"
    envBaseParams, envSpecificParams, modelParams, options, state_columns, action_columns =\
                            loadParameters(args.env_id, hpPath, args.HPfilename, algorithm)

    sweep_config['parameters'] = modelParams
    # sweep_config['envBaseParams'] = envBaseParams
    # sweep_config['envSpecificParams'] = envSpecificParams
    # sweep_config['options'] = options
    # sweep_config['state_columns'] = state_columns
    # sweep_config['action_columns'] = action_columns

    modelParams.update(parameter_dict)
    fixed_model_params = {
        'policy': {
            'value': 'MlpPolicy'
            },
        'n_steps': {
            'value': 128
            },
        'batch_size':{
            'value': 64
            },
        'n_epochs':{
            'value': 10
            },
        'gamma':{
            'value': 0.99
            },
        'gae_lambda':{
            'value': 0.95
            },
        'clip_range':{
            'value': 0.2
            },
        'normalize_advantage':{
            'value': True
            },
        'ent_coef':{
            'value': 0.0
            },
        'vf_coef':{
            'value': 0.5
            },
        'max_grad_norm':{
            'value': 0.5
            },
        'use_sde':{
            'value': False
            },
        'sde_sample_freq':{
            'value': -1
            },
        'target_kl':{
            'value': 'null'
            },

        'policy_kwargs':{ 
            'value': {
                'net_arch': {
                    'pi': [32, 32], 'vf': [256, 256]
                    },
                'optimizer_class': 'ADAM',
                'optimizer_kwargs': {'amsgrad': True},
                'activation_fn': 'Tanh'
                }
            },
    }

    modelParams.update(fixed_model_params)
    sweep_id = wandb.sweep(sweep_config, project=args.project)

    wandb.agent(sweep_id, train, count=1)
