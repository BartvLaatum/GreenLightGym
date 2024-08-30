import os
import argparse
from multiprocessing import cpu_count
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from greenlight_gym.experiments.utils import load_env_params, load_model_params, wandb_init, make_vec_env, create_callbacks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightHeatCO2")
    parser.add_argument("--project", type=str, default="testing")
    parser.add_argument("--group", type=str, default="group1")
    parser.add_argument("--config_name", type=str, default="four_controls")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--num_cpus", type=int, default=12)
    parser.add_argument("--n_evals", type=int, default=10)
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()

    # check cpus available
    assert args.num_cpus <= cpu_count(), \
        f"Number of CPUs requested ({args.num_cpus}) is greater than available ({cpu_count()})"

    env_config_path = f"configs/envs/"
    model_config_path = f"configs/algorithms/"

    states2plot = ["Air Temperature","CO2 concentration", "Humidity", "Fruit harvest", "PAR", "Cumulative harvest", "Cumulative CO2", "Cumulative gas", "Cumulative profit", "Cumulative violations"]
    actions2plot = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp"]
    
    algorithm = "ppo"
    env_base_params, env_specific_params, options, state_columns, action_columns = load_env_params(args.env_id, env_config_path, args.config_name)
    model_params = load_model_params(algorithm, model_config_path, args.config_name)



    monitor_filename = None
    vec_norm_kwargs = {"norm_obs": True, "norm_reward": True, "clip_obs": 50_000}

    env = make_vec_env(
        args.env_id,
        env_base_params,
        env_specific_params,
        options,
        seed=args.seed,
        n_envs=args.num_cpus,
        monitor_filename=monitor_filename,
        vec_norm_kwargs=vec_norm_kwargs
        )

    eval_env = make_vec_env(
        args.env_id,
        env_base_params,
        env_specific_params,
        options,
        seed=args.seed,
        n_envs=min(args.n_eval_episodes, args.num_cpus),
        monitor_filename=monitor_filename,
        vec_norm_kwargs=vec_norm_kwargs,
        eval_env=True,
        )
    print(eval_env)