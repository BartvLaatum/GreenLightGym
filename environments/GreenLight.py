import gymnasium as gym
from gymnasium.spaces import Box

import numpy as np
import pandas as pd
from typing import Optional, Any, Tuple
from RLGreenLight.environments.GreenLightCy import GreenLight as GL
from RLGreenLight.environments.pyutils import loadWeatherData, satVp

from datetime import date

class GreenLight(gym.Env):
    """
    Python wrapper class for the GreenLight model implemented in Cython.
    As input we get the weather data, number of variables for states, inputs and disturbances, and whether and which lamps we use.
    Argument definitions are in the init function.
    """
    def __init__(self, weatherDataDir:str, # path to weather data
                 location: str,             # location of the recorded weather data
                 dataSource: str,           # source of the weather data
                 h: float,                  # [s] time step for the RK4 solver
                 nx: int,                   # number of states 
                 nu: int,                   # number of control inputs
                 nd: int,                   # number of disturbances
                 noLamps: int,              # whether lamps are used
                 ledLamps: int,             # whether led lamps are used
                 hpsLamps: int,             # whether hps lamps are used
                 seasonLength: int,         # [days] length of the growing season
                 predHorizon: int,          # [days] number of future weather predictions
                 timeinterval: int,         # [s] time interval in between observations
                 controlSignals: list[str],  # list with all the control signals we aim to with RL/controller
                 modelObsVars: int,          # number of variables we observe from the model
                 weatherObsVars: int,        # number of variables we observe from the weather data
                 obsLow: list[float],       # lower bound for the observation space
                 obsHigh: list[float],      # upper bound for the observation space
                 rewardCoefficients: list[float], # coefficients for the reward function
                 penaltyCoefficients: list[float], # coefficients for the penalty function
                 options: Optional[dict[str, Any]] = None, # options for the environment (e.g. specify starting date)
                 training: bool = True,     # whether we are training or testing
                 ) -> None:
        super(GreenLight, self).__init__()
        # number of seconds in the day
        c = 86400

        # arguments that are kept the same over various simulations
        self.weatherDataDir = weatherDataDir
        self.location = location
        self.dataSource = dataSource
        self.h = h
        self.nx = nx
        self.nu = nu
        self.nd = nd
        self.noLamps = noLamps
        self.ledLamps = ledLamps
        self.hpsLamps = hpsLamps
        self.seasonLength = seasonLength
        self.predHorizon = predHorizon
        self.timeinterval = timeinterval
        self.controlSignals = controlSignals
        self.N = int(seasonLength*c/timeinterval)           # number of timesteps to take for python wrapper
        self.solverSteps = int(self.timeinterval/self.h)    # number of steps the solver takes between timeinterval
        self.Np = int(predHorizon*c/timeinterval)           # the prediction horizon in timesteps for our weather predictions
        self.modelObsVars = modelObsVars
        self.weatherObsVars = weatherObsVars
        self.rewardCoefficients = rewardCoefficients
        self.penaltyCoefficients = penaltyCoefficients
        self.training = training
        self.options = options

        # set up the action and observation space
        self.action_space = Box(low=-1, high=1, shape=(len(controlSignals),), dtype=np.float32)
        self.observation_space = Box(low=-1e4, high=1e4, shape=(self.modelObsVars+(self.Np+1)*self.weatherObsVars,), dtype=np.float32)

        # specify which signals we want to control, fixed various simulations
        controlIndices = {"boiler": 0, "co2": 1, "thermal": 2, "roofvent": 3, "lamps": 4, "intlamps": 5, "growpipes": 6, "blackout": 7}
        self.controlIdx = np.array([controlIndices[controlInput] for controlInput in controlSignals], dtype=np.int8)

        # lower and upper bounds for air temperature, co2 concentration, humidity
        self.obsLow = np.array(obsLow)
        self.obsHigh = np.array(obsHigh)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Given an action computed by some agent, we simulate the next state of the system.
        The system is modelled by the GreenLight model implemented in Cython.
        We have to power to control the boiler valve, co2, thermal screen, roof vent, lamps, internal lamps, grow pipes and blackout screen.
        """
        # scale the action to the range of the control inputs, which is between [0, 1]
        action = (action-self.action_space.low)/(self.action_space.high-self.action_space.low)
        self.GLModel.step(action, self.controlIdx)

        # state = self.GLModel.getStatesArray()
        obs = self.getObs()
        if self.terminalState(obs):
            self.terminated = True

        reward = self.rewardGrowth(obs, action)
        self.prevYield = obs[3]
        info = {}
        info = {"controls": self.GLModel.getControlsArray(),
                "Time": self.GLModel.time}

        return (obs,
            reward, 
            self.terminated, 
            False,
            info
            )

    def rewardGrowth(self, obs: np.ndarray, action: np.ndarray) -> float:
        deltaYield = obs[3] - self.prevYield
        reward = np.dot([deltaYield, *-action], self.rewardCoefficients)
        penalty = np.dot(self.computePenalty(obs), self.penaltyCoefficients)
        return reward - penalty

    def rewardFunction(self, obs: np.ndarray, action: np.ndarray) -> float:
        """
        Compute the reward for a given observation.
        Reward consists of the harvest over the past time step minus the costs of the control inputs.
        Reward reflect the profit of the greenhouse and constraint violations.
        """
        # make weighted sum from costs and harves
        # costs are negative, harvest is positive
        # harvest is in the obs[4] variable
        # costs are result from actions
        # print("previous Yield", self.prevYield)
        # print("current Yield", obs[3])
        # print(obs[3] - self.prevYield)
        reward = np.dot([obs[3], *-action], self.rewardCoefficients)


        penalty = np.dot(self.computePenalty(obs), self.penaltyCoefficients)
        return reward - penalty

    def computePenalty(self, obs: np.ndarray) -> float:
        # function to compute the penalty for the reward function
        # penalty is the sum of the squared differences between the observation and the bounds
        # penalty is zero if the observation is within the bounds
        Npen = self.obsLow.shape[0]
        lowerbound = self.obsLow[:] - obs[:Npen]
        lowerbound[lowerbound < 0] = 0
        upperbound = obs[:Npen] - self.obsHigh[:]
        upperbound[upperbound < 0] = 0
        return lowerbound**2 + upperbound**2

    def getObs(self) -> np.ndarray:
        # save co2 air, temperature air, humidity air, cFruit, par above the canpoy as effect from lamps and sun
        # retrieve par above canopy:
        modelObs = self.GLModel.getObs()
        weatherIdx = [self.GLModel.timestep] + [int(ts * self.timeinterval/self.h) + self.GLModel.timestep for ts in range(1, self.Np+1)]
        weatherObs = self.weatherData[weatherIdx, :self.weatherObsVars].flatten()

        return np.concatenate([modelObs, weatherObs], axis=0)

    def terminalState(self, states: np.ndarray) -> bool:
        if self.GLModel.timestep >= self.N:
            return True
        # check for nan and inf in state values
        elif np.isnan(states).any() or np.isinf(states).any():
            print("Nan or inf in states")
            return True
        return False

    def getTimeInDays(self) -> float:
        """
        Get time in days since 01-01-0001 upto the starting day of the simulation.
        """
        d0 = date(1, 1, 1)
        d1 = date(self.growthYear, 1, 1)
        delta = d1 - d0
        return delta.days + self.startDay

    def reset(self, seed: int | None = None, options: dict[str: Any]=None) -> Tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to a random initial state.
        Randomness is introduced by the growth year and start day.
        Which affect the weather data observed.
        """
        super().reset(seed=seed)
        if self.training:
            self.growthYear = np.random.choice([year for year in range(2012, 2020)])
            self.startDay = np.random.choice([day for day in range(274, 334)])
        else:
            self.growthYear = self.options["growthYear"]
            self.startDay = self.options["startDay"]

        # load in weather data for this simulation
        self.weatherData = loadWeatherData(self.weatherDataDir, self.location, self.dataSource, self.growthYear, self.startDay, self.seasonLength, self.predHorizon, self.h, self.nd)
        self.actions = np.zeros((self.nu,))

        # compute days since 01-01-0000
        # as time indicator by the model
        timeInDays = self.getTimeInDays()
        self.GLModel = GL(self.weatherData, self.h, self.nx, self.nu, self.nd, self.noLamps, self.ledLamps, self.hpsLamps, self.solverSteps, timeInDays)
        self.terminated = False
        obs = self.getObs()
        self.prevYield = obs[3]
        obs[[4,5]] = 0
        return obs, {}

def runRuleBasedController(GL, options):
    obs, info = GL.reset(options=options)
    N = GL.N                                        # number of timesteps to take
    states = np.zeros((N+1, GL.modelObsVars))       # array to save states
    controlSignals = np.zeros((N+1, GL.GLModel.nu)) # array to save rule-based controls controls
    states[0, :] = obs[:GL.modelObsVars]             # get initial states
    timevec = np.zeros((N+1,))                      # array to save time
    timevec[0] = GL.GLModel.time
    i=1
    while not GL.terminated:
        controls = np.ones((GL.action_space.shape[0],))*0.5
        # convert controls to np.float32
        # controls = controls.astype(np.float32)

        obs, r, terminated, _, info = GL.step(controls.astype(np.float32))
        states[i, :] += obs[:GL.modelObsVars]
        controlSignals[i, :] += info["controls"]
        timevec[i] = info["Time"]
        i+=1
    # insert time vector into states array
    states = np.insert(states, 0, timevec, axis=1)
    states = pd.DataFrame(data=states[:], columns=["Time", "Air Temperature", "CO2 concentration", "Humidity", "Fruit weight", "Fruit harvest", "PAR"])
    controlSignals = pd.DataFrame(data=controlSignals, columns=["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uIntLamp", "uGroPipe", "uBlScr"])
    weatherData = pd.DataFrame(data=GL.weatherData[[int(ts * GL.timeinterval/GL.h) for ts in range(0, GL.Np+1)], :GL.weatherObsVars], columns=["Temperature", "Humidity", "PAR", "CO2 concentration", "Wind"])

    return states, controlSignals, weatherData
def runSimulationDefinedControls(GL, matlabControls, stateNames, matlabStates, nx):
    obs, info = GL.reset()
    N = matlabControls.shape[0]

    cythonStates = np.zeros((N, nx))
    cyhtonControls = np.zeros((N, GL.GLModel.nu))
    cythonStates[0, :] = GL.GLModel.getStatesArray()

    for i in range(1, N):
        # print(i)
        controls = matlabControls.iloc[i, :].values
        obs, reward, terminated, truncated, info = GL.step(controls)
        cythonStates[i, :] += GL.GLModel.getStatesArray()
        cyhtonControls[i, :] += info["controls"]
        # print("Day of the year", GL.GLModel.dayOfYear)
        # print("time since midnight in hours", GL.GLModel.timeOfDay)
        # print("Time lamp of the day", GL.GLModel.lampTimeOfDay)

        if terminated:
            break

    cythonStates = pd.DataFrame(data=cythonStates, columns=stateNames[:])
    cyhtonControls = pd.DataFrame(data=cyhtonControls, columns=["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uIntLamp", "uGroPipe", "uBlScr", "shScr", "perShScr", "uSide"])
    return cythonStates, cyhtonControls

if __name__ == "__main__":
    pass