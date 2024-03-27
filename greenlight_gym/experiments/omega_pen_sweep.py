from argparse import ArgumentParser

import numpy as np

from greenlight_gym.common.results import Results
from greenlight_gym.experiments.train_agent import runExperiment
from greenlight_gym.experiments.utils import load_env_params, load_model_params

# STATE = {"tem/p": 0, "co2": 1, "rh": 2}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightHeatCO2")
    parser.add_argument("--env_config_name", type=str, default="multiplicative_pen")
    parser.add_argument("--project", type=str, default="omega_sweep")
    parser.add_argument("--group", type=str, default="omega")
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--start_range", type=float, default=0)
    parser.add_argument("--end_range", type=float, default=1e-3)
    parser.add_argument("--n_values", type=int, default=11)
    parser.add_argument("--n_evals", type=int, default=10)
    parser.add_argument("--num_cpus", type=int, default=12)
    parser.add_argument("--n_eval_episodes", type=int, default=60)
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--SEED", type=int, default=666)
    args = parser.parse_args()

    env_config_path = f"configs/envs/"
    model_config_path = f"configs/algorithms/"

    env_base_params, env_specific_params, options, result_columns = load_env_params(args.env_id, env_config_path, args.env_config_name)
    model_params = load_model_params(args.algorithm, model_config_path, args.env_config_name)
    results = Results(result_columns)

    omegas = np.linspace(args.start_range, args.end_range, args.n_values)
    print(omegas)
    # idx = STATE[args.group]
    for i, omega in enumerate(omegas):
        run_name = f"k-{args.group}-{(omega)}-{args.SEED}"

        env_specific_params["omega"] = omega
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
