import pandas as pd

path = "data/newFuncArgs/"
states1 = f"old-states20111001-120.csv"
states2 = f"new-states20111001-120.csv"

states1 = pd.read_csv(path+states1, sep=",")
states2 = pd.read_csv(path+states2, sep=",")

print(states1.head())
print(states2.head())

numericCols = states1.columns[1:]
print((states1[numericCols].values-states2[numericCols].values).sum())

# plt.plot(states1.iloc[:,1:])s