"""
This script contains the function to train an agent on the GreenLight environment.
It logs the training results to Weights and Biases (wandb) and saves the best model based on the evaluation results.
You can run this script from the command line with the following command:
    $ python -m greenlight.experiments.train_agent
There are multiple command line arguments you can use to customize the training process.
"""

import os
import argparse
from multiprocessing import cpu_count
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor

from greenlight_gym.experiments.utils import load_env_params, load_model_params, wandb_init, make_vec_env, create_callbacks, make_env
from greenlight_gym.common.results import Results

def runExperiment(env_id,
                  env_base_params,
                  env_specific_params,
                  options,
                  model_params,
                  env_seed,
                  model_seed,
                  n_eval_episodes,
                  num_cpus, 
                  project,
                  group,
                  total_timesteps,
                  n_evals,
                  results,
                  runname = None,
                  job_type="train",
                  save_model=True,
                  save_env=True
                  ) -> None:
    """
    Function that creates the training and evaluation environments, and the RL-model.
    It then trains the model on the environment and logs the results to wandb.
    Every eval_freq timesteps the model is evaluated on the evaluation environment.
    The best model based on the evaluation results is saved. Including env normalisation metrics.

    Args:
        env_id (str): environment id
        env_base_params (Dict): arguments for the base environment (GreenLightEnv)
        env_specific_params (Dict): arguments for the specific environment (GreenLightHeatEnv)
        options (Dict): additional options for the environment
        model_params (Dict): Hyperparameters for the RL model
        env_seed (int): random seed for the environment
        model_seed (int): random seed for the RL model
        n_eval_episodes (int): number of episodes to evaluate the agent for
        num_cpus (int): number of parrallel environments to use
        project (str): wandb project to log the training stats
        group (str): wandb group the run belongs to
        total_timesteps (int): number of timesteps to train the model for
        n_evals (int): number of evaluations to perform during training
        results (Results): module to store the results of the evaluation environment
        runname (str, optional): how to run the name, otherwise wandb makes one. Defaults to None.
        job_type (str, optional): job_type for wandb. Defaults to "train".
        save_model (bool, optional): whether to save the RL model. Defaults to True.
        save_env (bool, optional): whether to save the environment. Defaults to True.
    """

    run, config = wandb_init(
            model_params,
            env_base_params,
            env_specific_params,
            total_timesteps,
            env_seed,
            model_seed,
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
        env_base_params,
        env_specific_params,
        options,
        seed=env_seed,
        n_envs=num_cpus,
        monitor_filename=monitor_filename,
        vec_norm_kwargs=vec_norm_kwargs
        )

    eval_env = make_vec_env(
        env_id,
        env_base_params,
        env_specific_params,
        options,
        seed=env_seed,
        n_envs=min(n_eval_episodes, 10),
        monitor_filename=monitor_filename,
        vec_norm_kwargs=vec_norm_kwargs,
        eval_env=True,
        )

    if not runname:
        runname = run.name
    # exit()

    if save_model:
        model_log_dir = f"train_data/{project}/models/{runname}/"
    else:
        model_log_dir = None

    if save_env:
        env_log_dir = f"train_data/{project}/envs/{runname}/"
    else:
        env_log_dir =None

    eval_freq = total_timesteps//n_evals//num_cpus
    save_name = "vec_norm"

    callbacks = create_callbacks(
        n_eval_episodes,
        eval_freq,
        env_log_dir,
        save_name,
        model_log_dir,
        eval_env,
        run=run,
        results=results,
        save_env=save_env,
        verbose=1
        )

    tensorboard_log = f"train_data/{project}/logs/{runname}"

    if args.continue_training:

        env = SubprocVecEnv([make_env(env_id, rank, env_seed, env_base_params, env_specific_params, options, eval_env=False) for rank in range(num_cpus)])
        env = VecMonitor(env, filename=monitor_filename)

        model_path = f"train_data/{args.continued_project}/models/{args.continued_runname}/best_model.zip"
        env_path = f"train_data/{args.continued_project}/envs/{args.continued_runname}"

        # if I am correct we don't have to do this for the eval env, since it its synced during the evaluation callback.
        env = VecNormalize.load(os.path.join(env_path, "vecnormalize.pkl"), env)
        model = PPO.load(model_path, env=env, tensorboard_log=tensorboard_log, **config["model_params"])
        print(model.ent_coef)
    else:
        model = PPO(
            env=env,
            seed=model_seed,
            verbose=0,
            **config["model_params"],
            tensorboard_log=tensorboard_log
            )

    # run.log
    model.learn(total_timesteps=config["total_timesteps"], 
                callback=callbacks, reset_num_timesteps=False)

    # also save the final model configuration
    model.save(os.path.join(model_log_dir, "last_model"))
    env_save_path = os.path.join(env_log_dir, "last_vecnormalize.pkl")
    model.get_vec_normalize_env().save(env_save_path)

    # properly shutdown all the variables
    # and do garbage collection
    run.finish()
    env.close()
    eval_env.close()
    del model
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightHeatCO2", help="Environment ID")
    parser.add_argument("--project", type=str, default="testing", help="Wandb project name")
    parser.add_argument("--group", type=str, default="group1", help="Wandb group name")
    parser.add_argument("--env_config_name", type=str, default="four_controls", help="Name of the environment config file")
    parser.add_argument("--total_timesteps", type=int, default=500_000, help="Total number of timesteps to train algorithm for")
    parser.add_argument("--n_eval_episodes", type=int, default=1, help="Number of episodes to evaluate the agent for")
    parser.add_argument("--num_cpus", type=int, default=12, help="Number of CPUs to use during training")
    parser.add_argument("--n_evals", type=int, default=10, help="Number times we evaluate algorithm during training")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm to use")
    parser.add_argument("--env_seed", type=int, default=666, help="Random seed for the environment for reproducibility")
    parser.add_argument("--model_seed", type=int, default=666, help="Random seed for the RL-model for reproducibility")
    parser.add_argument('--save_model', default=True, action=argparse.BooleanOptionalAction, help="Whether to save the model")
    parser.add_argument("--save_env", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the environment")
    parser.add_argument("--continue_training", default=False, action=argparse.BooleanOptionalAction, help="Continue training from a saved model")
    parser.add_argument("--continued_project", type=str, default=None, help="Project name of the saved model to continue training from")
    parser.add_argument("--continued_runname", type=str, default=None, help="Runname of the saved model to continue training from")
    args = parser.parse_args()

    # check cpus available
    assert args.num_cpus <= cpu_count(), \
        f"Number of CPUs requested ({args.num_cpus}) is greater than available ({cpu_count()})"

    env_config_path = f"greenlight_gym/configs/envs/"
    model_config_path = f"greenlight_gym/configs/algorithms/"

    env_base_params, env_specific_params, options, result_columns = load_env_params(args.env_id, env_config_path, args.env_config_name)
    model_params = load_model_params(args.algorithm, model_config_path, args.env_config_name)
    results = Results(result_columns)

    job_type = f"seed-{args.model_seed}"
    runExperiment(args.env_id,
                    env_base_params,
                    env_specific_params, 
                    options,
                    model_params,
                    args.env_seed,
                    args.model_seed,
                    args.n_eval_episodes,
                    args.num_cpus, 
                    args.project,
                    args.group,
                    args.total_timesteps,
                    args.n_evals,
                    results=results,
                    runname=None,
                    job_type=job_type
                    )
