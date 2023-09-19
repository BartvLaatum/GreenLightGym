import pandas as pd
from RLGreenLight.visualisations import createFigs
from matplotlib import pyplot as plt
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20111001", help="Starting date of the simulation")
    parser.add_argument("--seasonLength", type=int, default=120, help="Length of the season")
    parser.add_argument("--controller", type=str, default="ppo", help="Controller to compare")
    parser.add_argument("--runname",  type=str, help="Runname of the controller")
    parser.add_argument("--months", nargs="*", type=int, default=[10], help="Month to plot")
    parser.add_argument("--days", nargs="*", type=int, default=None, help="Days to plot")
    args = parser.parse_args()

    # load data rule based controller
    baselineStates = pd.read_csv(f"data/ruleBasedControl/states{args.date}-{args.seasonLength:03}.csv")
    baselineControls = pd.read_csv(f"data/ruleBasedControl/controls{args.date}-{args.seasonLength:03}.csv")

    # load ppo data
    ppoStates = pd.read_csv(f"data/{args.controller}/{args.runname}/states{args.date}-{args.seasonLength:03}.csv")
    ppoControls = pd.read_csv(f"data/{args.controller}/{args.runname}/controls{args.date}-{args.seasonLength:03}.csv")

    states2plot = baselineStates.columns[1:]
    controls2plot = ppoControls.columns[:]

    # extract first of october from time column in baselineStates dataframe abd next five days
    baselineStates["Time"] = pd.to_datetime(baselineStates["Time"])
    ppoStates["Time"] = pd.to_datetime(ppoStates["Time"])
    baselineControls["Time"] = baselineStates["Time"]
    ppoControls["Time"] = ppoStates["Time"]

    baselineStates = baselineStates[baselineStates["Time"].dt.month.isin(args.months)]
    ppoStates = ppoStates[ppoStates["Time"].dt.month.isin(args.months)]
    # baselineStates = baselineStates.reset_index(drop=True)

    # extract data from first of october and november
    baselineControls = baselineControls[baselineControls["Time"].dt.month.isin(args.months)]
    ppoControls = ppoControls[ppoControls["Time"].dt.month.isin(args.months)]
    if args.days:
        baselineControls = baselineControls[baselineControls["Time"].dt.day.isin(args.days)]
        baselineStates = baselineStates[baselineStates["Time"].dt.day.isin(args.days)]
        ppoStates = ppoStates[ppoStates["Time"].dt.day.isin(args.days)]	
        ppoControls = ppoControls[ppoControls["Time"].dt.day.isin(args.days)]

    # plot the data
    fig, axes = createFigs.createStatesFig(states2plot)
    fig, axes = createFigs.plotVariables(fig, axes, baselineStates, states2plot, "Rule-based controller", "C00")
    fig, axes = createFigs.plotVariables(fig, axes, ppoStates, states2plot, "PPO", "C01")
    axes[0].legend()
    plt.show()

    fig, axes = createFigs.createStatesFig(controls2plot)
    fig, axes = createFigs.plotVariables(fig, axes, baselineControls, controls2plot, "Rule-based controller", "C00")
    fig, axes = createFigs.plotVariables(fig, axes, ppoControls, controls2plot, "PPO", "C01")
    axes[0].legend()
    plt.show()
