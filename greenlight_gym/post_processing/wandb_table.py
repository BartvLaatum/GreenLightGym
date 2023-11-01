import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df =pd.read_csv("data/k-barrier-tuning/co2-20110301.csv")
df["number"] = df["name"].str.extract('(\d+)').astype(int)
df['Group'] = ((df['number']-1)//5)+1

aggregated = df.groupby(["Group", "Time"]).agg({"CO2 concentration": ["mean", "std"]}).reset_index()
aggregated.columns = [' '.join(col).strip() for col in aggregated.columns.values]

print(df.head())
print(aggregated)

unique_groups = aggregated["Group"].unique()

colors = plt.cm.Greens(np.linspace(0,1, len(unique_groups)))
fig, ax = plt.subplots(figsize=(10,6))

for i, group in enumerate(unique_groups):
    subset = aggregated[aggregated["Group"]==group]
    ax.plot(subset["Time"], subset["CO2 concentration mean"], label=f"Group {group}", color=colors[i])
    # ax.fill_between(subset["Time"],
    #                 subset["CO2 concentration mean"] - subset["CO2 concentration std"],
    #                 subset["CO2 concentration mean"] + subset["CO2 concentration std"],
    #                 alpha=0.2)
plt.legend()
plt.show()
