from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cmcrameri.cm as cmc

from greenlight_gym.visualisations.createFigs import create_state_fig, plot_aggregated_trajectories

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

def aggregate_runs(df: pd.DataFrame, groups: List[str], statistics: List[str], states2plot: List[str], n_runs: int) -> pd.DataFrame:
    """Aggregate runs with the same hyperparameters

    Args:
        df (pd.DataFrame): dataframe with runs
        groups (List): values to group by
        statistics (List): statistics to compute
        states2plot (List): columns to aggregate
        n_runs (int): number of runs to aggregate
    Returns:
        pd.DataFrame: dataframe with aggregated runs
    """
    # group each n experiments, where n == n_runs
    df["number"] = df["runname"].str.extract('(\d+)').astype(int)
    df['Group'] = ((df['number']-1)//n_runs) + 1

    aggregated = df.groupby(groups).agg({state: statistics for state in states2plot}).reset_index()
    aggregated.columns = [' '.join(col).strip() for col in aggregated.columns.values]
    return aggregated

def indoor_vars_per_k(aggregated_df, k_factors, states2plot, unique_groups, lower_bounds=None, upper_bounds=None, p_bands=None):
    colors = cmc.batlow(np.linspace(0,1, len(unique_groups)))
    # Define discrete boundaries for the colors
    boundaries = np.linspace(-5, -3, N+1)
    norm = mpl.colors.BoundaryNorm(boundaries, cmc.batlow.N, clip=True)

    # Create a scalar mappable object
    sm = plt.cm.ScalarMappable(cmap=cmc.batlow, norm=norm)
    sm.set_array([])  # Dummy array for the ScalarMappable

    fig, axes = create_state_fig(time, 
                            states2plot,
                            lower_bounds,
                            upper_bounds,
                            p_bands)

    for i, ax in enumerate(axes):
        ax = plot_aggregated_trajectories(ax, aggregated_df, unique_groups, states2plot[i], labels=k_factors, colors=colors)

    # Add the colorbar
    cbar = fig.colorbar(sm, ticks=boundaries, ax=axes)
    cbar.set_label(r'$k_{CO_2}$')
    cbar.ax.set_yticklabels([f"{10**val:.1e}" for val in boundaries])

    return fig, axes

df = pd.read_csv("data/k-barrier-tuning/arctan-penalty/20110301-states.csv")
df['Time'] = pd.to_datetime(df['Time'], unit='ms')
time = df["Time"].unique()

N = 13
k_factors = np.logspace(-5,-3, N)
states2plot = ["Air Temperature","CO2 concentration", "Humidity"]
n_runs = 5
statistics = ["mean", "std"]
aggregated_df = aggregate_runs(df, ["Group", "Time"], statistics, states2plot, n_runs)

unique_groups = aggregated_df["Group"].unique()
upper_bounds = [34, 1000, 85]
lower_bounds = [15, 300, 60]
p_bands = [0.5, 100, 2]

fig, axes = indoor_vars_per_k(aggregated_df, k_factors, states2plot, unique_groups, lower_bounds, upper_bounds, p_bands)

# compute time outside of bounds
df["CO2 out of bounds"] = np.where((df["CO2 concentration"] > upper_bounds[1]) | (df["CO2 concentration"] < lower_bounds[1]), 1, 0)

aggregate_df = aggregate_runs(df, ["Group"], ["sum"], ["CO2 out of bounds"], n_runs)

# compute percentage of time outside of bounds
aggregate_df["CO2 out of bounds mean"] = aggregate_df["CO2 out of bounds sum"] / df["Group"].value_counts().values

# create bar chart of percentage of time outside of bounds
fig = plt.figure(dpi=120)
fig.text(x=0.05, y=0.5, s="Frequency out of CO2 bounds/p-band (%)", rotation=90, va="center", ha="center")

ax = fig.add_subplot(2, 1, 1)
colors = cmc.batlow(np.linspace(0,1, len(unique_groups)))
ax.bar(aggregate_df["Group"], aggregate_df["CO2 out of bounds mean"], color=colors)
ax.set_ylim([0, 0.8])

# dont show xticks
ax.set_xticks([])
plt.grid(visible=False)

df["CO2 out of p-band"] = np.where((df["CO2 concentration"] > upper_bounds[1]+p_bands[1]) | (df["CO2 concentration"] < lower_bounds[1]), 1, 0)
aggregate_df = aggregate_runs(df, ["Group"], ["sum"], ["CO2 out of p-band"], n_runs)

# compute percentage of time outside of bounds
aggregate_df["CO2 out of p-band mean"] = aggregate_df["CO2 out of p-band sum"] / df["Group"].value_counts().values

# create bar chart of percentage of time outside of bounds
# fig, ax = plt.subplots()
ax = fig.add_subplot(2, 1, 2)

ax.bar(aggregate_df["Group"], aggregate_df["CO2 out of p-band mean"], color=colors)
ax.set_xlabel(r'$k_{CO_2}$')

ax.set_xticks(aggregate_df["Group"])
ax.set_xticklabels([f"{val:.1e}" for val in k_factors[:]], rotation=45, ha="right")
ax.set_ylim([0, 0.8])
plt.grid(visible=False)
plt.tight_layout()
plt.show()

statistics = ["mean", "std"]
states2plot = ["Cumulative harvest"]
aggregated_df = aggregate_runs(df, ["Group", "Time"], statistics, states2plot, n_runs)
fig, axes = indoor_vars_per_k(aggregated_df, k_factors, states2plot, unique_groups)
plt.show()
# compute harvest from each run
