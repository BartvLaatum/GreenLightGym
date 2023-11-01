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

# def bin_values(sub_df):
#     # Define the bin edges
#     bins = [i/10 for i in range(12)]
    
#     # Bin the values
#     sub_df['bin'] = pd.cut(sub_df['uCO2'], bins=bins, right=False, labels=bins[:-1])
    
#     # Compute the frequencies
#     return sub_df['bin'].value_counts().sort_index()/len(sub_df)

def bin_values(sub_df):
    # Define the bin edges
    bins = [i/10 for i in range(12)]
    
    # Bin the values
    labels = [i/10 for i in range(11)]
    sub_df['bin'] = pd.cut(sub_df['uCO2'], bins=bins, right=False, labels=labels)
    
    # Return the entire DataFrame with the 'bin' column
    return sub_df
def classify_day_night(time):
    if (time >= pd.to_datetime("01:00")) and (time < pd.to_datetime("18:00")):
        return 'day'
    else:
        return 'night'

df = pd.read_csv("data/k-barrier-tuning/co2-test/20110301-controls.csv")
df['Time'] = pd.to_datetime(df['Time'], unit='ms')
time = df["Time"].unique()

n_runs = 5
df["number"] = df["runname"].str.extract('(\d+)').astype(int)
df['Group'] = ((df['number']-1)//n_runs) + 1
df['day_night'] = df['Time'].apply(classify_day_night)

colors = cmc.batlow(np.linspace(0,1, df['Group'].nunique()))
# Define discrete boundaries for the colors
boundaries = np.linspace(-4, -2, df['Group'].nunique())
norm = mpl.colors.BoundaryNorm(boundaries, cmc.batlow.N, clip=True)

# Create a scalar mappable object
sm = plt.cm.ScalarMappable(cmap=cmc.batlow, norm=norm)
sm.set_array([])  # Dummy array for the ScalarMappable

# result = df.groupby('Group').apply(bin_values).reset_index()
# fig, ax = plt.subplots()
# for group in result['Group']:
#     # Extract the row for the group
#     row = result[result['Group'] == group]
    
#     # Drop the 'Group' column and calculate the cumulative sum
#     cumulative_frequencies = row.drop(columns=['Group']).cumsum(axis=1)
    
#     # Normalize to get relative frequencies
#     total = cumulative_frequencies.iloc[:, -1].values[0]
#     relative_frequencies = cumulative_frequencies / total

#     # Plot
#     relative_frequencies.iloc[0].plot(ax=ax, label=group, color=colors[group-1], linewidth=3)

# # Format the plot
# # Add the colorbar
# cbar = fig.colorbar(sm, ticks=boundaries, ax=ax)
# cbar.set_label(r'$k_{CO_2}$')
# cbar.ax.set_yticklabels([f"{10**val:.0e}" for val in boundaries])

# ax.set_xlabel(r'Relative CO$_2$ injection')
# ax.set_ylabel('Cumulative')
# # ax.set_title('Cumulative Plot of Relative Frequencies')
# # ax.legend(title='Group')
# ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# # Display the plot
# plt.tight_layout()
# plt.show()

print(df.head())
binned_values = df.groupby(['Group', 'day_night']).apply(bin_values)
frequencies = binned_values.groupby(['Group', 'day_night', 'bin']).size().unstack().fillna(0).astype(int)
# Initialize a new plot
fig, ax = plt.subplots()
print(frequencies)
# Calculate and plot cumulative relative frequencies for each group and day/night classification
for group in frequencies.index.get_level_values('Group').unique():
    for dn in ['day', 'night']:
        subset = frequencies.loc[(group, dn)]
        cumulative_frequencies = subset.cumsum()
        total = cumulative_frequencies.iloc[-1]
        relative_frequencies = cumulative_frequencies / total
        label = f"{group} ({dn})"
        relative_frequencies.plot(ax=ax, label=label)

# Format the plot
ax.set_xlabel('Binned Values')
ax.set_ylabel('Relative Cumulative Frequency')
ax.set_title('Cumulative Plot of Relative Frequencies (Day vs Night)')
ax.legend(title='Group (Day/Night)')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Display the plot
plt.tight_layout()
plt.show()