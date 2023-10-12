import pandas as pd
from RLGreenLight.visualisations import createFigs
from matplotlib import pyplot as plt
import argparse
import os
import cmcrameri.cm as cmc
import numpy as np

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
    dmfm = 0.0627
    tomatoPrice = 1.2
    co2price = 0.1
    timeinterval = 900
    co2availble = 72000/14000
    states['Cumulative reward'] = (states['Fruit harvest']/dmfm * tomatoPrice - controls["uCO2"] * co2availble* 1e-6 *  timeinterval * co2price).cumsum()
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

    ppoStatesResults = []
    ppoControlResults = []
    labels = []

    if args.runname == "all":
        # extract all runnames from ppo folder
        runnames = os.listdir(f"data/{args.controller}/{args.project}/")
        #load in the data from all runs in pd dataframes and save in list
        for runname in runnames[:]:
            labels.append(runname)
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
    

    states2plot = ["Air Temperature", "CO2 concentration", "PAR", "Cumulative reward"]
    controls2plot = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp"]

    ylabels = ["Air\n temperature [$^\circ$C]", "CO2\n concentration [ppm]","PAR [W/m$^2$]", "Cumulative\n profit [EUR/m$^2$]"]

    # extract first of october from time column in baselineStates dataframe abd next five days
    baselineStates["Time"] = pd.to_datetime(baselineStates["Time"])
    baselineControls["Time"] = pd.to_datetime(baselineStates["Time"])
    colorgrad = np.linspace(0, 1, len(ppoControlResults)+1)

    # plot the data
    fig, axes = createFigs.createStatesFig(ylabels)
    # fig, axes = createFigs.plotVariables(fig, axes, baselineStates, states2plot, "Rule-based controller", cmc.grayCS(0))

    createFigs.setStateBounds(axes[0], 14, 34)
    createFigs.setStateBounds(axes[1], 300, 1200)
    # createFigs.setStateBounds(axes[2], 60, 90)

    for i, ppoStates in enumerate(ppoStatesResults):
        fig, axes = createFigs.plotVariables(fig, axes, ppoStates, states2plot, labels[i], cmc.grayCS(i+1))
    axes[0].legend(bbox_to_anchor=(1.0, 0.25), loc='upper left', borderaxespad=0.)
    plt.show()

    fig, axes = createFigs.createStatesFig(controls2plot)
    # fig, axes = createFigs.plotVariables(fig, axes, baselineControls, controls2plot, "Rule-based controller", cmc.grayCS(0))
    for i, ppoControls in enumerate(ppoControlResults):
        fig, axes = createFigs.plotVariables(fig, axes, ppoControls, controls2plot, labels[i], cmc.grayCS(i+1))

    # plot legend in lower right corner
    axes[-1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)
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
