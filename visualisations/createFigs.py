import seaborn as sns#; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from RLGreenLight.environments.pyutils import days2date
import matplotlib.dates as mdates

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
# show grid
plt.rcParams['axes.grid'] = True


def createStatesFig(states2plot: list):
    """
    Function to plot variables simulated by GL model.
    """
    fig = plt.figure(dpi=120)
    nplots = len(states2plot)

    if nplots % 2 == 0:
        rows = nplots//2
        cols = 2
    else:
        rows = nplots//2 + 1
        cols = nplots//2 + 1
    # add subplot for each state
    axes = [fig.add_subplot(rows, cols, i+1) for i in range(nplots)]

    # set the xlabel for the final row of plots
    # remove xticks for all but the bottom row
    xlabel = "Time (days)"
    for ax in axes[:-2]:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xlabel(xlabel)

    # set the ylabel for each plot
    for i, ax in enumerate(axes):
        ax.set_ylabel(states2plot[i])
    fig.tight_layout()
    # plt.show()
    return fig, axes

def setStateBounds(ax, lowerBound, upperBound):
    """
    Function to set the bounds of the state axes.
    """
    # plot line for upper and lower bound
    ax.axhline(lowerBound, color="grey", linestyle="--", alpha=0.5, linewidth=3)
    ax.axhline(upperBound, color="grey", linestyle="--", alpha=0.5, linewidth=3)
    return ax

def plotVariables(fig, axes: list, states: pd.DataFrame, states2plot: list, label: str, color: str):
    """
    Function to plot variables simulated by GL model.
    """
    for i, ax in enumerate(axes):
        ax.step(states["Time"], states[states2plot[i]], label=label, color=color, alpha=0.8, linewidth=3)
    # rotate xticks
    for ax in axes:
        # ax.tick_params(axis='x', rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()
    return fig, axes

def createDistFig(states2plot: list):
    """
    Function to plot variables simulated by GL model.
    """
    fig = plt.figure()
    nplots = len(states2plot)
    # add subplot for each state
    axes = [fig.add_subplot(nplots//2, 2, i+1) for i in range(nplots)]
    
    # set the xlabel for the final row of plots
    # remove xticks for all but the bottom row
    # xlabel = "Time (days)"
    for i, ax in enumerate(axes[:-2]):
        # ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xlabel(states2plot[i])

    # set the ylabel for each plot
    for i, ax in enumerate(axes):
        ax.set_ylabel("Count")
    return fig, axes


def plotDistributions(fig, axes: list, states: pd.DataFrame, states2plot: list, label: str, color: str):
    """
    Function to plot variables simulated by GL model.
    """
    for i, ax in enumerate(axes):
        sns.histplot(states[states2plot[i]], ax=ax, label=label, color=color)
    return fig, axes

# def createBarPlot(states2plot: list):
    # """

    # axes = states.plot(x="Time", y=states2plot, subplots=True, linewidth=3, alpha=0.8, color=color, label=[label]*len(states2plot))

    # for ax in axes[1:]:
    #     ax.legend().remove()
    # # set set ylabels to column names
    # for i, ax in enumerate(axes):
    #     ax.set_ylabel(states2plot[i])
    # axes.title(title)
    # return axes
