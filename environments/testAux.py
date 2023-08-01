"""
We use this file to test our GreenLight implementation in cython.

Creates a GreenLight object in cython.
"""
# from __future__ import print_function
from GreenLightCy import GreenLight
import numpy as np
import pandas as pd
from pyutils import loadWeatherData
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load weather csv file
    weatherDataMat = pd.read_csv('../data/weatherMatlab.csv', sep=',', names=['iGlob', 'tOut', 'vpOut', 'co2Out', 'wind', 'tSky', 'tSoOut', 'radSum'])

    # load control data
    controlDataMat = pd.read_csv("../data/controlsMatlab.csv", sep=",", names=["lamp", "intLamp", "boil", "boilGro", "tSo"])

    weatherData = loadWeatherData()

    # load states data
    stateDataMat = pd.read_csv("../data/statesMatlab.csv", sep=",", names=["Time", "co2Air", "co2Top", "tAir", "tTop", "tCan", "tCovIn", "tCovE", "tThScr", \
                                                                            "tFlr", "tPipe", "tSo1", "tSo2", "tSo3", "tSo4", "tSo5", "vpAir", "vpTop", "tLamp", \
                                                                            "tIntLamp", "tGroPipe", "tBlScr", "tCan24", "cBuf", "cLeaf", "cStem", "cFruit", "tCanSum"])
    # x = np.arange(0, weatherDataMat.shape[0], 1)

    # # plot weather data in subplots
    # plt.figure()
    # plt.subplot(4,2,1)
    # plt.plot(weatherDataMat["iGlob"], label="iGlob")
    # plt.plot(x, weatherData[x,0], linestyle="--",  label='iGlob python')

    # plt.subplot(4,2,2)
    # plt.plot(weatherDataMat["tOut"], label="tOut")
    # plt.plot(x, weatherData[x,1], linestyle="--",  label='tOut python')

    # plt.subplot(4,2,3)
    # plt.plot(weatherDataMat["vpOut"], label="vpOut")
    # plt.plot(x, weatherData[x,2], linestyle="--",  label='vpOut python')

    # plt.subplot(4,2,4)
    # plt.plot(weatherDataMat["co2Out"], label="co2Out")
    # plt.plot(weatherData[x,3], linestyle="--",  label='co2Out python')

    # plt.subplot(4,2,5)
    # plt.plot(weatherDataMat["wind"], label="wind")
    # plt.plot(weatherData[x,4], linestyle="--",  label='wind python')

    # plt.subplot(4,2,6)
    # plt.plot(weatherDataMat["tSky"], label="tSky")
    # plt.plot(weatherData[x,5], linestyle="--",  label='tSky python')

    # plt.subplot(4,2,7)
    # plt.plot(weatherDataMat["tSoOut"], label="tSoOut")
    # plt.plot(weatherData[x,6], linestyle="--",  label='tSoOut python')

    # plt.legend()
    # # plt.show()

    # # plot both weather data
    # plt.figure()
    # plt.plot(x, weatherDataMat['vpOut'], label='vpOut data')
    # plt.legend()
    # plt.show()
    np.set_printoptions(suppress=True)
    h = 60 # sampling period [s] = 5 min
    noLamps, ledLamps, hpsLamps = 0, 1, 0
    GL = GreenLight(weatherData, h, noLamps, ledLamps, hpsLamps)
    N = 2
    testIndex = 0
    for i in range(N):
        print("------------------")
        print("step:", i)
        GL.step(testIndex)
        GL.setStates(stateDataMat.iloc[i+1, 1:].values, testIndex)
        print("python x:", GL.getStatesArray())
