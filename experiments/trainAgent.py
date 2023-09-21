from RLGreenLight.environments.GreenLight import GreenLightBase, GreenLightProduction
from RLGreenLight.experiments.utils import loadParameters, wandb_init, make_vec_env, create_callbacks
from stable_baselines3 import PPO
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import wandb

import argparse

envs = {"GreenLightBase": GreenLightBase, "GreenLightProduction": GreenLightProduction}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="TestVecLoadSave")
    parser.add_argument("--group", type=str, default="testing-evaluation")
    parser.add_argument("--HPfolder", type=str, default="GLBase")
    parser.add_argument("--HPfilename", type=str, default="ppo.yml")
    parser.add_argument("--env_id", type=str, default="GreenLightBase")
    parser.add_argument("--total_timesteps", type=int, default=400_000)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--numCpus", type=int, default=4)
    parser.add_argument("--n_evals", type=int, default=10)
    args = parser.parse_args()

    # check cpus available
    assert args.numCpus <= cpu_count(), f"Number of CPUs requested ({args.numCpus}) is greater than available ({cpu_count()})"

    hpPath = f"hyperparameters/{args.HPfolder}/"
    action_columns = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uIntLamp", "uGroPipe", "uBlScr"]
    state_columns = ["Air Temperature", "CO2 concentration", "Humidity", "Fruit weight", "Fruit harvest", "PAR", "Hour of the Day", "Day of the Year"]
    states2plot = ["CO2 concentration", "Fruit weight", "Fruit harvest", "PAR", "Cumulative harvest"]
    actions2plot = ["uCO2"]

    SEED = 666
    n_eval_episodes = 1
    algorithm = "PPO"
    envBaseParams, envSpecificParams, modelParams, options =\
                            loadParameters(args.env_id, hpPath, args.HPfilename, algorithm)

    # define
    run, config = wandb_init(envs[args.env_id],
                             modelParams,
                             envBaseParams,
                             envSpecificParams,
                             options,
                             args.total_timesteps,
                             SEED,
                             project=args.project,
                             group=args.group,
                             job_type="train",
                             save_code=True)
    
    monitor_filename = None
    vec_norm_kwargs = {"norm_obs": True, "norm_reward": True, "clip_obs": 50_000}

    env = make_vec_env(config["env"],
                       numCpus=args.numCpus,
                       monitor_filename=monitor_filename,
                       vec_norm_kwargs=vec_norm_kwargs)

    eval_env = make_vec_env(config["eval_env"],
                            numCpus=1,
                            monitor_filename=monitor_filename,
                            vec_norm_kwargs=vec_norm_kwargs,
                            eval_env=True)

    env_log_dir = f"trainData/{args.project}/envs/{run.name}/"
    model_log_dir = f"trainData/{args.project}/models/{run.name}/"
    eval_freq = args.total_timesteps//args.n_evals//args.numCpus
    save_name = "vec_norm"

    callbacks = create_callbacks(n_eval_episodes,
                                 eval_freq,
                                 env_log_dir,
                                 save_name,
                                 model_log_dir,
                                 eval_env,
                                 run=None,
                                 action_columns=action_columns,
                                 state_columns=state_columns,
                                 states2plot=states2plot,
                                 actions2plot=actions2plot,
                                 verbose=1)

    tensorboard_log = f"trainData/{args.project}/logs/{run.name}"
    model = PPO(env=env, seed=SEED, verbose=0, **config["modelParams"], tensorboard_log=tensorboard_log)
    model.learn(total_timesteps=config["total_timesteps"], 
                callback=callbacks)
    run.finish()
