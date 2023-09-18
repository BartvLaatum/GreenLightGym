import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from RLGreenLight.experiments.utils import loadParameters, wandb_init, make_vec_env, create_callbacks
from RLGreenLight.environments.GreenLight import GreenLight, runRuleBasedController
from RLGreenLight.visualisations.createFigs import createStatesFig, plotVariables
from RLGreenLight.environments.pyutils import days2date


def runNominalController(envParams, options):
    GL = GreenLight(**envParams, options=options, training=False)
    # time controller
    states, controls, weather = runRuleBasedController(GL, options)
    states["Time"] = np.asarray(days2date(states["Time"].values, "01-01-0001"), "datetime64[s]")
    controls["Time"] = states["Time"]
    weather["Time"] = states["Time"]
    return states, controls, weather

def interLightsExp(envParams, options):
    """
    Test the effect of controlling inter lights on the fruit weight.
    Currently, the inter lights are controlled by the rule based controller. 
    """
    noIntLights = runNominalController(envParams, options)
    envParams["intLamps"] = 1
    interLights = runNominalController(envParams, options)

    # plot states and controls
    # states2plot = noIntLights[0].columns[1:]
    vars2plot = ["Fruit weight", "uIntLamp"]

    noIntLights[0][vars2plot[0]]

    # create dataframe from states and controls
    noIntLightsDf = pd.DataFrame(columns=vars2plot)
    noIntLightsDf["Fruit weight"] = noIntLights[0]["Fruit weight"].to_numpy()
    noIntLightsDf["uIntLamp"] = noIntLights[1]["uIntLamp"].to_numpy()
    noIntLightsDf["Time"] = noIntLights[0]["Time"]

    # create dataframe from states and controls
    intLightsDf = pd.DataFrame(columns=vars2plot)
    intLightsDf["Fruit weight"] = interLights[0]["Fruit weight"].to_numpy()
    intLightsDf["uIntLamp"] = interLights[1]["uIntLamp"].to_numpy()
    intLightsDf["Time"] = interLights[0]["Time"]

    fig, axes = createStatesFig(vars2plot)

    plotVariables(fig, axes=axes, states=intLightsDf, states2plot=vars2plot, label="Inter lighting", color="C00")
    plotVariables(fig, axes=axes, states=noIntLightsDf, states2plot=vars2plot, label="No inter lights", color="C01")
    plt.legend()
    plt.show()

def testControlScheme(envParams, options, controlVar, nightValue, dayValue):
    GL = GreenLight(**envParams, options=options, training=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20111001", help="Starting date of the simulation")

    hpPath = "hyperparameters/modelDynamics/"
    filename = "interLamps.yml"
    env_id = "GreenLight"
    envParams, modelParams, options = loadParameters(env_id, hpPath, filename)
    interLightsExp(envParams, options)


