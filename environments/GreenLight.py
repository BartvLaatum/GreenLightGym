import gymnasium as gym
from gymnasium.spaces import Box

import yaml
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

from RLGreenLight.environments.GreenLightCy import GreenLight as GL
from RLGreenLight.environments.pyutils import loadMatlabData, loadWeatherData
from RLGreenLight.visualisations.compareModels import plotStates


class GreenLight(gym.Env):
    """
    Python wrapper class for the GreenLight model implemented in Cython.
    As input we get the weather data, number of variables for states, inputs and disturbances, and whether and which lamps we use.
    """
    def __init__(self, weatherDataDir:str,  # path to weather data
                 h: float,                  # [s] time step for the RK4 solver
                 nx: int,                   # number of states 
                 nu: int,                   # number of control inputs
                 nd: int,                   # number of disturbances
                 noLamps: int,              # whether lamps are used
                 ledLamps: int,             # whether led lamps are used
                 hpsLamps: int,             # whether hps lamps are used
                 seasonLength: int,         # [days] length of the growing season
                 predHorizon: int,          # [days] number of future weather predictions
                 startDay: int,             # start day of the growing season
                 timeinterval: int,         # [s] time interval in between observations
                 controlSignals: list[str]  # list with all the control signals we aim to with RL/controller (other e.g., rulebased TODO: IMPLEMENT THIS RULE-BASED CONTROLLER)
                 ) -> None:
        super(GreenLight, self).__init__()

        # number of seconds in the day
        c = 86400
        self.action_space = Box(low=-1, high=1, shape=(nu,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(nx,), dtype=np.float32)

        # specify which signals we want to control
        controlIndices = {"boiler": 0, "co2": 1, "thermal": 2, "roofvent": 3, "lamps": 4, "intlamps": 5, "growpipes": 6, "blackout": 7}
        self.controlIdx = [controlIndices[controlinput] for controlinput in controlSignals]
        self.actions = np.zeros((nu,))

        self.N = int(seasonLength*c/timeinterval)   # number of timesteps to take for python wrapper
        solverSteps = int(timeinterval/h)           # number of steps the solver takes between timeinterval

        # load in weather data for this simulation
        # also returns the future predictions
        self.weatherData, self.Np = loadWeatherData(weatherDataDir, startDay, seasonLength, predHorizon, h, nd)

        # initialise GL model in cython
        self.GLModel = GL(self.weatherData, self.Np, h, nx, nu, nd, noLamps, ledLamps, hpsLamps, solverSteps)

    def step(self, action: np.ndarray) -> tuple:

        self.actions[self.controlIdx] = action[self.controlIdx]
        self.GLModel.step(self.actions)
        state = self.GLModel.getStatesArray()

        if self.terminalState(state):
            self.terminated = True

        obs = self.getObs(state)
        reward = 0
        info = {}

        return (obs,
            reward, 
            self.terminated, 
            False,
            info
            )

    def getObs(self, state: np.ndarray) -> np.ndarray:
        # save co2 air, temperature air, humidity air and cFruit
        # retrieve par above canopy:

        return np.array([state[0], state[2], state[15], state[25], self.GLModel.rParGhSun, self.GLModel.rParGhLamp])

    def terminalState(self, states: np.ndarray) -> bool:
        if self.GLModel.timestep >= self.N:
            return True
        # check for nan and inf in state values
        elif np.isnan(states).any() or np.isinf(states).any():
            return True
        return False

    def reset(self):
        self.GLModel.reset()
        self.terminated = False

def runSimulation(Gl, matlabControls, stateNames, matlabStates, nx):
    GL.reset()
    N = matlabControls.shape[0]

    cythonStates = np.zeros((N, nx))
    cythonStates[0, :] = GL.GLModel.getStatesArray()

    for i in range(1, N):

        controls = matlabControls.iloc[i, :].values
        obs, reward, terminated, truncated, info = GL.step(controls)
        cythonStates[i, :] += GL.GLModel.getStatesArray()

        if terminated:
            break
    cythonStates = pd.DataFrame(data=cythonStates, columns=stateNames[1:])
    cythonStates["Time"] = matlabStates["Time"].values
    return cythonStates

if __name__ == "__main__":
    # load in yaml file from hyperparameters
    with open("hyperparameters/GreenLight.yml", "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)["GreenLight"]
    
    stateNames = ["Time", "co2Air", "co2Top", "tAir", "tTop", "tCan", "tCovIn", "tCovE", "tThScr", \
                "tFlr", "tPipe", "tSo1", "tSo2", "tSo3", "tSo4", "tSo5", "vpAir", "vpTop", "tLamp", \
                "tIntLamp", "tGroPipe", "tBlScr", "tCan24", "cBuf", "cLeaf", "cStem", "cFruit", "tCanSum"]


    # use the controls at 5 min stepsize
    matlabStates, matlabControls, _ = loadMatlabData(stepSize="5min", date="1Januari", stateNames=stateNames)
    GL = GreenLight(**params)
    cythonStates = runSimulation(GL, matlabControls, stateNames, matlabStates, params['nx'])


    plotStates(matlabStates, cythonStates, None, stateNames[3:10], "titled")