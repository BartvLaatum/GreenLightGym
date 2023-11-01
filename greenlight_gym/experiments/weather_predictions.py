import argparse

from greenlight_gym.experiments.train_agent import runExperiment
from greenlight_gym.experiments.utils import loadParameters

HPfolders = {"GreenLightHeatCO2": "GLHeatCO2"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id",type=str, default="GreenLightHeatCO2")
    args = parser.parse_args()

    HPfolder = HPfolders[args.env_id]
    hpPath = f"hyperparameters/{HPfolder}/"
    states2plot = ["Air Temperature","CO2 concentration", "Humidity", "Fruit harvest", "PAR", "Cumulative harvest"]
    actions2plot = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp"]

    SEED = 666
    algorithm = "PPO"
    envBaseParams, envSpecificParams, modelParams, options, state_columns, action_columns =\
                            loadParameters(args.env_id, hpPath, args.HPfilename, algorithm)

    runExperiment(
        args.env_id,
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
        states2plot=None,
        actions2plot=None,
        runname = None,
        )
