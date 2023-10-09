import time
import yaml
import argparse

import numpy as np
import seaborn as sns; sns.set()

from RLGreenLight.environments.pyutils import days2date
from RLGreenLight.experiments.utils import runRuleBasedController, make_env, loadParameters

def runNominalController(env_id, envParams, envSpecificParams, options, state_columns, action_columns):
    GL = make_env(env_id, rank=0, seed=666, kwargs=envParams, kwargsSpecific=envSpecificParams, options=options, eval_env=True)()
    # time controller
    start = time.time()
    states, controls, weather = runRuleBasedController(GL, state_columns, action_columns)
    end = time.time()
    print(f"Time to run controller: {end-start}")
    states["Time"] = np.asarray(days2date(states["Time"].values, "01-01-0001"), "datetime64[s]")
    controls["Time"] = states["Time"]
    weather["Time"] = states["Time"]

    # convert first date of time column to YYYYMMDD format
    dates = states["Time"].dt.strftime("%Y%m%d")

    states.to_csv(f"data/ruleBasedControl/states{dates[0]}-{envParams['seasonLength']:03}.csv", index=False)
    controls.to_csv(f"data/ruleBasedControl/controls{dates[0]}-{envParams['seasonLength']:03}.csv", index=False)
    weather.to_csv(f"data/ruleBasedControl/weather{dates[0]}-{envParams['seasonLength']:03}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GreenLightBase")
    parser.add_argument("--HPfolder", type=str, default="GLBase")
    parser.add_argument("--project", type=str, default="GLProduction")
    args = parser.parse_args()

    path = f"hyperparameters/{args.HPfolder}/"

    envBaseParams, envSpecificParams, modelParams, options, state_columns, action_columns = loadParameters(args.env_id, path, "ruleBasedControl.yml")
    runNominalController(args.env_id, envBaseParams, envSpecificParams, options, state_columns, action_columns)
