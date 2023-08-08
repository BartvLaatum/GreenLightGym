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
import argparse

if __name__ == "__main__":
    # argparse stepsize and euler
    parser = argparse.ArgumentParser(description='Run GreenLight in cython.')
    parser.add_argument('--stepsize', type=str, default="60s", help='Stepsize of the simulation')
    parser.add_argument('--euler', action="store_true", help='Euler integration, if not passed than RK4 is used.')
    args = parser.parse_args()

    datapath = f"data/matlab/{args.stepsize}StepSize"

    # load weather csv file
    weatherDataMat = pd.read_csv(datapath + "Weather.csv", sep=',', names=['iGlob', 'tOut', 'vpOut', 'co2Out', 'wind', 'tSky', 'tSoOut', 'radSum'])

    # load control data
    controlDataMat = pd.read_csv(datapath + "Controls.csv", sep=",", names=["boil", "extCo2", "thScr", "roof", "lamps", "intLamp", "boilGro", "blScr"])
    controlDataMat["shScr"] = 0.0 # add missing column
    controlDataMat["shScrPer"] = 0.0 # add missing column
    controlDataMat["side"] = 0.0 # add missing column

    # load states data
    stateNames = ["Time", "co2Air", "co2Top", "tAir", "tTop", "tCan", "tCovIn", "tCovE", "tThScr", \
                                                                            "tFlr", "tPipe", "tSo1", "tSo2", "tSo3", "tSo4", "tSo5", "vpAir", "vpTop", "tLamp", \
                                                                            "tIntLamp", "tGroPipe", "tBlScr", "tCan24", "cBuf", "cLeaf", "cStem", "cFruit", "tCanSum"]
    stateDataMat = pd.read_csv(datapath + "States.csv", sep=",", names=stateNames)

    # time vector in seconds
    time = stateDataMat["Time"].values
    # this is preprocessed weather data at a fixed 5 minute intervals
    weatherData = loadWeatherData()

    np.set_printoptions(suppress=True)
    noLamps, ledLamps, hpsLamps = 0, 1, 0 # which lamps do we use?
    nx, nu, nd = 27, 11, 7 # number of states, inputs and disturbances
    h = 3 # sampling period [s] = 3 seconds
    start = 0
    times = []

    GL = GreenLight(weatherDataMat.values[start:,:7], h, nx, nu, nd, noLamps, ledLamps, hpsLamps)
    N = stateDataMat.shape[0]
    cythonStates = np.zeros((stateDataMat.shape[0], stateDataMat.shape[1]-1))
    cythonStates[0, :] = GL.getStatesArray()
    tstart = t()

    for i in range(start, int(N/100)):

        # var time interval:
        controls = controlDataMat.iloc[i, :].values
        # GL.setSstates(stateDataMat.iloc[i, 1:].values, 0)
        GL.step(controls)
        cythonStates[i+1, :] += GL.getStatesArray()

    cythonStates = pd.DataFrame(data= cythonStates, columns=stateNames[1:])
    cythonStates["Time"] = time
    times += [t()-tstart]
    print(f"Average time elapsed: {np.mean(times)}, std: {np.std(times)}")

    # if args.euler:
    #     print("Euler")
    #     cythonStates.to_csv(f"data/cython/{args.stepsize}StepSizeStatesEuler.csv", index=False)
    # else:
    #     print("RK4"	)
    #     cythonStates.to_csv(f"data/cython/{args.stepsize}StepSizeStatesRK4.csv", index=False)
