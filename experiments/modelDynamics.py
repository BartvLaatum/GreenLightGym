import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from RLGreenLight.experiments.utils import loadParameters, wandb_init, make_vec_env, create_callbacks
from RLGreenLight.environments.GreenLight import GreenLight, runRuleBasedController, controlScheme
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

def runControlScheme(envParams, options, nightValue, dayValue):
    GL = GreenLight(**envParams, options=options, training=False)
    states, controls, weather = controlScheme(GL, nightValue, dayValue)
    states["Time"] = np.asarray(days2date(states["Time"].values, "01-01-0001"), "datetime64[s]")
    controls["Time"] = states["Time"]
    weather["Time"] = states["Time"]
    return states, controls, weather

def controlSchemeExp(envParams, options, controlVar):
    """
    Test the effect of controlling inter lights on the fruit weight.
    Currently, the inter lights are controlled by the rule based controller.
    """
    lowNight = runControlScheme(envParams, options, nightValue=-1, dayValue=-.8)
    highNight = runControlScheme(envParams, options, nightValue=-1, dayValue=-.8)

    vars2plot = ["CO2 concentration", "Fruit weight", "PAR"]
    lowNightdf = pd.DataFrame(columns=vars2plot)
    # lowNightdf[vars2plot[0]] = lowNight[1][vars2plot[0]].to_numpy()
    lowNightdf[vars2plot[0]] = lowNight[0][vars2plot[0]].to_numpy()
    lowNightdf[vars2plot[1]] = lowNight[0][vars2plot[1]].to_numpy()
    lowNightdf[vars2plot[2]] = lowNight[0][vars2plot[2]].to_numpy()
    lowNightdf["Time"] = lowNight[0]["Time"]

    highNightdf = pd.DataFrame(columns=vars2plot)
    # highNightdf[vars2plot[0]] = highNight[1][vars2plot[0]].to_numpy()
    highNightdf[vars2plot[0]] = highNight[0][vars2plot[0]].to_numpy()
    highNightdf[vars2plot[1]] = highNight[0][vars2plot[1]].to_numpy()
    highNightdf[vars2plot[2]] = highNight[0][vars2plot[2]].to_numpy()

    highNightdf["Time"] = highNight[0]["Time"]


    # plot controls
    fig, axes = createStatesFig(vars2plot)
    plotVariables(fig, axes=axes, states=lowNightdf, states2plot=vars2plot, label="Control", color="C00")
    plotVariables(fig, axes=axes, states=highNightdf, states2plot=vars2plot, label="Control", color="C01")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20111001", help="Starting date of the simulation")

    hpPath = "hyperparameters/modelDynamics/"
    filename = "interLamps.yml"
    env_id = "GreenLight"
    envParams, modelParams, options = loadParameters(env_id, hpPath, filename)
    # interLightsExp(envParams, options)

    filename = "co2Effects.yml"
    envParams, modelParams, options = loadParameters(env_id, hpPath, filename)

    controlVar = "uCO2"
    controlSchemeExp(envParams, options, controlVar)
