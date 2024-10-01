import argparse

import numpy as np

from greenlight_gym.experiments.train_agent import runExperiment
from greenlight_gym.experiments.utils import loadParameters

def set_k_factor(k, envSpecificParams):
    envSpecificParams["k"][0] = k
    return envSpecificParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightHeatCO2")
    parser.add_argument("--project", type=str, default="k-factor-barrier")
    parser.add_argument("--group", type=str, default="testing-evaluation")
    parser.add_argument("--HPfolder", type=str, default="GLHeatCO2")
    parser.add_argument("--HPfilename", type=str, default="ppo.yml")
    parser.add_argument("--total_timesteps", type=int, default=5100)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--numCpus", type=int, default=12)
    parser.add_argument("--n_evals", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=667)
    args = parser.parse_args()

    hpPath = f"greenlight_gym/hyperparameters/{args.HPfolder}/"
    states2plot = ["Air Temperature","CO2 concentration", "Humidity", "Fruit harvest", "PAR", "Cumulative harvest"]
    actions2plot = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp"]

    seeds = range(args.seed, args.seed+args.n_runs)

    algorithm = "PPO"
    envBaseParams, envSpecificParams, modelParams, options, state_columns, action_columns =\
                            loadParameters(args.env_id, hpPath, args.HPfilename, algorithm)

    k_co2 = np.logspace(-2, 0, 13)
    for k in k_co2[:]:
        group = f"k-factor={k:.5}"
        envSpecificParams = set_k_factor(k, envSpecificParams)
        print(envSpecificParams["k"])
        for run in range(args.n_runs):
            if run == 0:
                a2plot = actions2plot
                s2plot = states2plot
            else:
                a2plot = []
                s2plot = []
                
            runExperiment(args.env_id,
                            envBaseParams,
                            envSpecificParams, 
                            options,
                            modelParams,
                            seeds[run],
                            args.n_eval_episodes,
                            args.numCpus, 
                            args.project,
                            group,
                            args.total_timesteps,
                            args.n_evals,
                            state_columns,
                            action_columns,
                            states2plot=states2plot,
                            actions2plot=actions2plot,
                            runname=None,
                            save_model=False,
                            save_env=False,
                            )