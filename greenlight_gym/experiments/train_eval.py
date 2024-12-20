import os
import argparse
from multiprocessing import cpu_count
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from stable_baselines3 import PPO

from greenlight_gym.experiments.utils import load_env_params, load_model_params, wandb_init, make_vec_env, create_callbacks
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
        env_log_dir = f"greenlight_gym/train_data/{project}/envs/{runname}/"
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

    tensorboard_log = f"greenlight_gym/train_data/{project}/logs/{runname}"

    model = PPO(
        env=env,
        seed=model_seed,
        verbose=0,
        **config["model_params"],
        tensorboard_log=tensorboard_log
        )

    # run.log
    model.learn(total_timesteps=config["total_timesteps"], 
                callback=callbacks)

    # also save the final model configuration
    model.save(os.path.join(model_log_dir, "last_model"))

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
    args = parser.parse_args()

    # check cpus available
    assert args.num_cpus <= cpu_count(), \
        f"Number of CPUs requested ({args.num_cpus}) is greater than available ({cpu_count()})"

    env_config_path = f"greenlight_gym/configs/envs/"
    model_config_path = f"greenlight_gym/configs/algorithms/"

    env_base_params, env_specific_params, options, result_columns = load_env_params(args.env_id, env_config_path, args.env_config_name)
    eval_train_params = {'start_train_year': 2001, 'end_train_year': 2010, 'train_days': [59, 90, 120, 151, 181, 212, 243]}
    env_base_params.update(eval_train_params)

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