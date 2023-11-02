import os
import argparse
from multiprocessing import cpu_count
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import wandb
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from greenlight_gym.experiments.utils import loadParameters, wandb_init, make_vec_env, create_callbacks

def runExperiment(
    env_id,
    envBaseParams,
    envSpecificParams,
    options,
    modelParams,
    seed,
    n_eval_episodes,
    numCpus, 
    project,
    group,
    total_timesteps,
    n_evals,
    state_columns,
    action_columns,
    states2plot=None,
    actions2plot=None,
    runname = None,
    job_type="train",
    save_model=True,
    save_env=True
    ):

    run, config = wandb_init(
            modelParams,
            envBaseParams,
            envSpecificParams,
            total_timesteps,
            seed,
            project=project,
            group=group,
            runname=runname,
            job_type=job_type,
            save_code=True
            )

    monitor_filename = None
    vec_norm_kwargs = {"norm_obs": True, "norm_reward": True, "clip_obs": 50_000}

    env = make_vec_env(
        env_id,
        envBaseParams,
        envSpecificParams,
        options,
        seed=seed,
        numCpus=numCpus,
        monitor_filename=monitor_filename,
        vec_norm_kwargs=vec_norm_kwargs
        )

    eval_env = make_vec_env(
        env_id,
        envBaseParams,
        envSpecificParams,
        options,
        seed=seed,
        numCpus=1,
        monitor_filename=monitor_filename,
        vec_norm_kwargs=vec_norm_kwargs,
        eval_env=True,
        )

    if not runname:
        runname = run.name
    # exit()


    if save_model:
        model_log_dir = f"trainData/{project}/models/{runname}/"
    else:
        model_log_dir = None

    if save_env:
        env_log_dir = f"trainData/{project}/envs/{runname}/"
    else:
        env_log_dir =None

    eval_freq = total_timesteps//n_evals//numCpus
    save_name = "vec_norm"

    callbacks = create_callbacks(
        n_eval_episodes,
        eval_freq,
        env_log_dir,
        save_name,
        model_log_dir,
        eval_env,
        run=run,
        action_columns=action_columns,
        state_columns=state_columns,
        states2plot=states2plot,
        actions2plot=actions2plot,
        save_env=save_env,
        verbose=1
        )

    tensorboard_log = f"trainData/{project}/logs/{runname}"

    model = PPO(
        env=env,
        seed=seed,
        verbose=0,
        **config["modelParams"],
        tensorboard_log=tensorboard_log
        )

    # run.log
    model.learn(total_timesteps=config["total_timesteps"], 
                callback=callbacks)

    # properly shutdown all the variables
    # and do garbage collection
    run.finish()
    env.close()
    eval_env.close()
    del model
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightHeatCO2")
    parser.add_argument("--project", type=str, default="TestVecLoadSave")
    parser.add_argument("--group", type=str, default="group1")
    parser.add_argument("--HPfolder", type=str, default="gl_heat_co2")
    parser.add_argument("--HPfilename", type=str, default="ppo_4_controls.yml")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--numCpus", type=int, default=12)
    parser.add_argument("--n_evals", type=int, default=10)
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()

    # check cpus available
    assert args.numCpus <= cpu_count(), \
        f"Number of CPUs requested ({args.numCpus}) is greater than available ({cpu_count()})"

    hpPath = f"hyperparameters/{args.HPfolder}/"
    states2plot = ["Air Temperature","CO2 concentration", "Humidity", "Fruit harvest", "PAR", "Cumulative harvest"]
    actions2plot = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp"]

    algorithm = "PPO"
    envBaseParams, envSpecificParams, modelParams, options, state_columns, action_columns =\
                            loadParameters(args.env_id, hpPath, args.HPfilename, algorithm)

    job_type = f"seed-{args.seed}"
    runExperiment(args.env_id,
                    envBaseParams,
                    envSpecificParams, 
                    options,
                    modelParams,
                    args.seed,
                    args.n_eval_episodes,
                    args.numCpus, 
                    args.project,
                    args.group,
                    args.total_timesteps,
                    args.n_evals,
                    state_columns,
                    action_columns,
                    states2plot=states2plot,
                    actions2plot=actions2plot,
                    runname=None,
                    job_type=job_type
                    )