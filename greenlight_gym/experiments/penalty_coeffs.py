from argparse import ArgumentParser

import numpy as np

from greenlight_gym.common.results import Results
from greenlight_gym.experiments.train_agent import runExperiment
from greenlight_gym.experiments.utils import load_env_params, load_model_params

STATE = {"temp": 0, "co2": 1, "rh": 2}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightHeatCO2")
    parser.add_argument("--env_config_name", type=str, default="5min_four_controls")
    parser.add_argument("--project", type=str, default="penalty-coeffs")
    parser.add_argument("--group", type=str, default="co2")
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--start_range", type=float, default=0)
    parser.add_argument("--end_range", type=float, default=1e-3)
    parser.add_argument("--n_values", type=int, default=11)
    parser.add_argument("--n_evals", type=int, default=1)
    parser.add_argument("--num_cpus", type=int, default=12)
    parser.add_argument("--n_eval_episodes", type=int, default=60)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--SEED", type=int, default=666)
    args = parser.parse_args()

    env_config_path = f"greenlight_gym/configs/envs/"
    model_config_path = f"greenlight_gym/configs/algorithms/"

    env_base_params, env_specific_params, options, result_columns = load_env_params(args.env_id, env_config_path, args.env_config_name)
    model_params = load_model_params(args.algorithm, model_config_path, args.env_config_name)
    results = Results(result_columns)

    coefficients = np.round(np.linspace(args.start_range, args.end_range, args.n_values), 5)

    idx = STATE[args.group]
    for i, ki in enumerate(coefficients):

        run_name = f"k-{args.group}-{(ki)}-{args.SEED}"
        env_specific_params["k"][idx] = ki
        runExperiment(args.env_id,
                        env_base_params,
                        env_specific_params,
                        options,
                        model_params,
                        args.SEED,
                        args.n_eval_episodes,
                        args.num_cpus, 
                        args.project,
                        args.group,
                        args.total_timesteps,
                        args.n_evals,
                        results,
                        runname = run_name,
                        job_type="train",
                        save_model=True,
                        save_env=True
                        )
