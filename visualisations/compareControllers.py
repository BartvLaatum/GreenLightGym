import pandas as pd
from RLGreenLight.visualisations import createFigs
from matplotlib import pyplot as plt
import argparse
import os
import cmcrameri.cm as cmc

### Latex font in plots
plt.rcParams['font.serif'] = "cmr10"
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.size'] = 24

plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc('axes', unicode_minus=False)


def computeReward(states, controls):
    tomatoPrice = 1.2
    co2price = 0.19
    timeinterval = 900
    co2availble = 72000/14000
    states['Cumulative reward'] = (states['Fruit harvest'] * tomatoPrice - controls["uCO2"] * co2availble* 1e-6 *  timeinterval * co2price).cumsum()
    return states

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20111001", help="Starting date of the simulation")
    parser.add_argument("--seasonLength", type=int, default=120, help="Length of the season")
    parser.add_argument("--controller", type=str, default="ppo", help="Controller to compare")
    parser.add_argument("--runname", type=str, help="Runname of the controller")
    parser.add_argument("--project", type=str, default="GLProduction", help="Project name")
    parser.add_argument("--months", nargs="*", type=int, default=[10], help="Month to plot")
    parser.add_argument("--days", nargs="*", type=int, default=[1, 2, 3, 4 ,5], help="Days to plot")
    args = parser.parse_args()

    # load data rule based controller
    baselineStates = pd.read_csv(f"data/ruleBasedControl/states{args.date}-{args.seasonLength:03}.csv")
    baselineControls = pd.read_csv(f"data/ruleBasedControl/controls{args.date}-{args.seasonLength:03}.csv")
    baselineStates.insert(6, "Cumulative Harvest", baselineStates["Fruit harvest"].cumsum()) 
    baselineStates= computeReward(baselineStates, baselineControls)
    # baselineStates["Cumulative Harvest"] = baselineStates["Fruit harvest"].cumsum()

    ppoStatesResults = []
    ppoControlResults = []
    labels = ['No Constraint', "CO2 constraint"]

    if args.runname == "all":
        # extract all runnames from ppo folder
        runnames = os.listdir(f"data/{args.controller}/{args.project}/")
        #load in the data from all runs in pd dataframes and save in list
        for runname in runnames[:]:
            # load ppo data
            ppoStates = pd.read_csv(f"data/{args.controller}/{args.project}/{runname}/states{args.date}-{args.seasonLength:03}.csv")
            # insert cumulative harvest column at beginning of dataframe
            ppoStates.insert(6, "Cumulative Harvest", ppoStates["Fruit harvest"].cumsum()) 
            ppoStates["Time"] = pd.to_datetime(ppoStates["Time"])

            ppoControls = pd.read_csv(f"data/{args.controller}/{args.project}/{runname}/controls{args.date}-{args.seasonLength:03}.csv")
            ppoControls["Time"] = pd.to_datetime(ppoStates["Time"])
            ppoStates= computeReward(ppoStates, ppoControls)

            ppoStatesResults.append(ppoStates)
            ppoControlResults.append(ppoControls)

    else:
        # load ppo data
        ppoStates = pd.read_csv(f"data/{args.controller}/{args.project}/{args.runname}/states{args.date}-{args.seasonLength:03}.csv")
        # insert cumulative harvest column at beginning of dataframe
        ppoStates.insert(6, "Cumulative Harvest", ppoStates["Fruit harvest"].cumsum()) 
        ppoStates["Time"] = pd.to_datetime(ppoStates["Time"])

        ppoControls = pd.read_csv(f"data/{args.controller}/{args.project}/{args.runname}/controls{args.date}-{args.seasonLength:03}.csv")
        ppoControls["Time"] = pd.to_datetime(ppoStates["Time"])
        ppoStates= computeReward(ppoStates, ppoControls)

        ppoStatesResults.append(ppoStates)
        ppoControlResults.append(ppoControls)
    

    states2plot = ["Air Temperature", "CO2 concentration", "Humidity", "Cumulative Harvest", "PAR", "Cumulative reward"] #+ ["Cumulative Harvest"]
    controls2plot = ["uCO2", "uLamp", "uVent"]

    # extract first of october from time column in baselineStates dataframe abd next five days
    baselineStates["Time"] = pd.to_datetime(baselineStates["Time"])
    baselineControls["Time"] = pd.to_datetime(baselineStates["Time"])
    import numpy as np
    colorgrad = np.linspace(0, 1, len(ppoControlResults)+1)

    # plot the data
    fig, axes = createFigs.createStatesFig(states2plot)
    fig, axes = createFigs.plotVariables(fig, axes, baselineStates, states2plot, "Rule-based controller", cmc.batlowS(0))
    for i, ppoStates in enumerate(ppoStatesResults):
        fig, axes = createFigs.plotVariables(fig, axes, ppoStates, states2plot, labels[i], cmc.batlowS(i+1))
    axes[0].legend()
    plt.show()

    fig, axes = createFigs.createStatesFig(controls2plot)
    fig, axes = createFigs.plotVariables(fig, axes, baselineControls, controls2plot, "Rule-based controller", cmc.batlowS(0))
    for i, ppoControls in enumerate(ppoControlResults):
        fig, axes = createFigs.plotVariables(fig, axes, ppoControls, controls2plot, labels[i], cmc.batlowS(i+1))
    axes[0].legend()
    plt.show()


    # baselineStates = baselineStates[baselineStates["Time"].dt.month.isin(args.months)]
    # ppoStates = ppoStates[ppoStates["Time"].dt.month.isin(args.months)]

    # # extract data from specified month
    # baselineControls = baselineControls[baselineControls["Time"].dt.month.isin(args.months)]
    # ppoControls = ppoControls[ppoControls["Time"].dt.month.isin(args.months)]
    # if args.days:
    #     baselineControls = baselineControls[baselineControls["Time"].dt.day.isin(args.days)]
    #     baselineStates = baselineStates[baselineStates["Time"].dt.day.isin(args.days)]
    #     ppoStates = ppoStates[ppoStates["Time"].dt.day.isin(args.days)]	
    #     ppoControls = ppoControls[ppoControls["Time"].dt.day.isin(args.days)]
