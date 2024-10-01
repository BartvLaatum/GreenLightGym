import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# import argparse
import numpy as np
from matplotlib.patches import Patch
# from greenlight_gym.common.utils import co2ppm2dens


def load_data(step_size, variable_names, var_type="States"):
    matlab_variables = pd.read_csv(f"data/model_comparison/matlab/{step_size}StepSize{var_type}.csv", sep=",", header=None)
    python_variables = pd.read_csv(f"data/model_comparison/python/{step_size}StepSize{var_type}.csv", sep=",")
    matlab_variables.columns = variable_names
    return matlab_variables, python_variables

if __name__ == "__main__":
    step_size = "1s"
    state_names = ["co2Air", "co2Top", "tAir", "tTop", "tCan", "tCovIn", "tCovE", "tThScr", \
            "tFlr", "tPipe", "tSo1", "tSo2", "tSo3", "tSo4", "tSo5", "vpAir", "vpTop", "tLamp", \
            "tIntLamp", "tGroPipe", "tBlScr", "tCan24", "cBuf", "cLeaf", "cStem", "cFruit", "tCanSum", "Time"]
    weather_names = ["Global radiation", "Outdoor temperature", "Outdoor VP", "Outdoor CO2 concentration", "Outdoor wind speed", "Sky temperature", "Soil temperature", "Daily radiation sum", "Daytime", "Daytime smoothed"]
    var_type = "States"
    matlab_states, python_states = load_data(step_size, state_names, var_type=var_type)

    print(matlab_states)
    print(python_states)

    var_type = "Weather"
    matlab_weather, python_weather = load_data(step_size, weather_names, var_type=var_type)
    
    # plot global radation difference
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(python_weather["glob_rad"], label="Python")
    ax.plot(matlab_weather["Global radiation"], label="Matlab")
    ax.set_ylabel("Global radiation")
    ax.set_xlabel("Time")
    ax.legend()
    plt.show()
