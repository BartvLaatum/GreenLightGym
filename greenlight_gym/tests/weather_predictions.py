from greenlight_gym.envs.greenlight import GreenLightBase

kwargs = {
    "weatherDataDir": "greenlight_gym/envs/data/", # path to the weather data file
    "location": "Amsterdam",   # location of the weatherdata
    "dataSource": "KNMI",   # which data source we use
    "nx": 28,   # Number of states
    "nu": 8,   # number of control variables for GreenLight
    "nd": 10,   # weather variables
    "noLamps": 0,   # whether lamps are used
    "ledLamps": 1,   # whether led lamps are used
    "hpsLamps": 0,   # whether hps lamps are used
    "intLamps": 0,   # whether int lamps are used
    "dmfm": 0.0627,   # dry matter fresh matter ratio
    "h": 1,   # stepsize for the RK4 solver
    "seasonLength": 1,   # number of growing days
    "predHorizon": 0,   # prediction horizon in days.
    "timeinterval": 900,   # [s] time interval at what rate do we observe and control the environment
    "controlSignals": ["uBoil", "uCO2", "uThScr", "uVent"], # list with all control inputs we aim to control/learn
    "modelObsVars": 6,         # number of variables we observe from the model
    "weatherObsVars": 5,       # number of weather variables we observe
    "obsLow":  [15, 300, 60],  # lower bound for the observation space
    "obsHigh": [35, 1000, 85], # upper bound for the observation space
    "training": True,
}

GL = GreenLightBase(**kwargs)
print(GL.observation_space.shape)
obs, info = GL.reset()
try:
    assert GL.observation_space.shape == obs.shape
except AssertionError:
    print("Assertion error, spaces are not aligned")
    print(GL.observation_space.shape, obs.shape)