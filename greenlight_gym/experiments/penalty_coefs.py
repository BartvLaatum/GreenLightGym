import numpy as np

from greenlight_gym.experiments.train_agent import runExperiment
from greenlight_gym.envs.greenlight import GreenLightCO2
from greenlight_gym.experiments.utils import loadParameters

if __name__ == "__main__":
    env_id  = "GreenLightCO2"
    numCpus = 12
    HPfolder = "GLCO2"
    HPfilename = "ppo.yml"
    project = "Penalty-coeffs"
    algorithm="PPO"
    group = "test"

    hpPath = f"hyperparameters/{HPfolder}/"

    total_timesteps = 1_000_000
    n_evals = 10
    n_eval_episodes = 1
    SEED = 666
    co2_coefficients = np.logspace(-5, -6, 1)
    envBaseParams, envSpecificParams, modelParams, options, state_columns, action_columns =\
                        loadParameters(env_id, hpPath, HPfilename, algorithm)
    envSpecificParams["reward"] = "scaled"
    for i, co2_c in enumerate(co2_coefficients):
        # envBaseParams["penaltyCoefficients"][1] = 1e-5
        group = f"reward-scaled"
        runExperiment(
            env_id,
            envBaseParams,
            envSpecificParams, 
            options,
            modelParams,
            SEED,
            n_eval_episodes,
            numCpus, 
            project,
            group,
            total_timesteps,
            n_evals,
            state_columns,
            action_columns,
            states2plot=state_columns,
            actions2plot=["uCO2"],
            runname=None,
        )
