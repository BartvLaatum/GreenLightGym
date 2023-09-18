import yaml
from RLGreenLight.environments.GreenLight import GreenLight, runRuleBasedController
from RLGreenLight.environments.pyutils import days2date
import seaborn as sns; sns.set()
import numpy as np
import time

def runNominalController(params, options):
    GL = GreenLight(**params, options=options, training=False)
    # time controller
    start = time.time()
    states, controls, weather = runRuleBasedController(GL, options)
    end = time.time()
    print(f"Time to run controller: {end-start}")
    states["Time"] = np.asarray(days2date(states["Time"].values, "01-01-0001"), "datetime64[s]")
    controls["Time"] = states["Time"]
    weather["Time"] = states["Time"]

    # convert first date of time column to YYYYMMDD format
    dates = states["Time"].dt.strftime("%Y%m%d")

    states.to_csv(f"data/ruleBasedControl/states{dates[0]}-{params['seasonLength']:03}.csv", index=False)
    controls.to_csv(f"data/ruleBasedControl/controls{dates[0]}-{params['seasonLength']:03}.csv", index=False)
    weather.to_csv(f"data/ruleBasedControl/weather{dates[0]}-{params['seasonLength']:03}.csv", index=False)

if __name__ == "__main__":
    with open("hyperparameters/ruleBasedControl.yml", "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        fixedParams = params["GreenLight"]
        options = params["options"]

    runNominalController(fixedParams, options)
