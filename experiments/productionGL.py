import time
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from RLGreenLight.experiments.utils import loadParameters, runRuleBasedController
from RLGreenLight.environments.GreenLight import GreenLightCO2
from RLGreenLight.environments.pyutils import days2date
from RLGreenLight.visualisations.createFigs import createStatesFig, plotVariables 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20111001", help="Starting date of the simulation")

    hpPath = "hyperparameters/ProductionGL/"
    filename = "ruleBased.yml"
    env_id = "GreenLightCO2"
    stateColumns = ["Time", "Air Temperature", "CO2 concentration", "Humidity", "Fruit weight", "Fruit harvest", "PAR", "Hour of the Day", "Day of the Year"]
    actionColumns = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uIntLamp", "uGroPipe", "uBlScr"]

    envBaseParams, envSpecificParams, modelParams, options = loadParameters(env_id, hpPath, filename)
    
    GL = eval(env_id)(**envSpecificParams, **envBaseParams, options=options, training=False)
    start = time.time()
    states, controls, weatherData = runRuleBasedController(GL, options, stateColumns, actionColumns)
    end = time.time()
    print(end-start)

    states2plot = ["Air Temperature", "CO2 concentration", "Fruit weight", "Cumulative harvest", "Humidity", "PAR"]
    states["Time"] = pd.to_datetime(days2date(timeInDays = states["Time"], referenceDate="01-01-0001"))
    states["Cumulative harvest"] = states["Fruit harvest"].cumsum()

    controls2plot = controls.columns
    controls["Time"] = states["Time"]

    fig, axes = createStatesFig(states2plot)
    plotVariables(fig, axes, states, states2plot, label="Production", color="C00")
    fig, axes = createStatesFig(controls2plot)
    plotVariables(fig, axes, controls, controls2plot, label="Production", color="C00")


    
    plt.show()

