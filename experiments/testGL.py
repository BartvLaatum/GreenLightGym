"""
We use this file to test our GreenLight implementation in cython.

Creates a GreenLight object in cython.
"""
# from __future__ import print_function
from RLGreenLight.environments.GreenLightCy import GreenLight
from time import time as t
import numpy as np
import pandas as pd
from RLGreenLight.environments.pyutils import loadWeatherData
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load weather csv file
    weatherDataMat = pd.read_csv('data/varStepSizeWeather.csv', sep=',', names=['iGlob', 'tOut', 'vpOut', 'co2Out', 'wind', 'tSky', 'tSoOut', 'radSum'])

    # load control data
    controlDataMat = pd.read_csv("data/varStepSizecontrols.csv", sep=",", names=["boil", "extCo2", "thScr", "roof", "lamps", "intLamp", "boilGro", "blScr"])
    controlDataMat["shScr"] = 0.0 # add missing column
    controlDataMat["shScrPer"] = 0.0 # add missing column
    controlDataMat["side"] = 0.0 # add missing column
    # load states data
    stateNames = ["Time", "co2Air", "co2Top", "tAir", "tTop", "tCan", "tCovIn", "tCovE", "tThScr", \
                                                                            "tFlr", "tPipe", "tSo1", "tSo2", "tSo3", "tSo4", "tSo5", "vpAir", "vpTop", "tLamp", \
                                                                            "tIntLamp", "tGroPipe", "tBlScr", "tCan24", "cBuf", "cLeaf", "cStem", "cFruit", "tCanSum"]
    stateDataMat = pd.read_csv("data/varStepSizeStates.csv", sep=",", names=stateNames)

    # time vector in seconds
    time = stateDataMat["Time"].values
    # this is preprocessed weather data at a fixed 5 minute intervals
    weatherData = loadWeatherData()

    np.set_printoptions(suppress=True)
    h = 1 # sampling period [s] = 5 min
    noLamps, ledLamps, hpsLamps = 0, 1, 0
    nx, nu, nd = 27, 11, 7

    times = []
    # for _ in range(100):
    GL = GreenLight(weatherDataMat.values[:,:7], h, nx, nu, nd, noLamps, ledLamps, hpsLamps)
    N = stateDataMat.shape[0]
    testIndex = 0
    cythonStates = np.zeros((stateDataMat.shape[0]-1, stateDataMat.shape[1]-1))

    tstart = t()
    for i in range(N-1):
        # var time interval:
        h = time[i+1] - time[i]
        controls = controlDataMat.iloc[i, :].values

        GL.step(testIndex, controls, h)
        cythonStates[i, :] += GL.getStatesArray()
    cythonStates = pd.DataFrame(data= cythonStates, columns=stateNames[1:])
    cythonStates["Time"] = time[:-1]

    times += [t()-tstart]

    print(f"Average time elapsed: {np.mean(times)}, std: {np.std(times)}")

