import pandas as pd
from scipy.signal import resample
from scipy.io import loadmat
import numpy as np
from RLGreenLight.environments.pyutils import loadWeatherData, rh2vaporDens, vaporDens2pres, co2ppm2dens, soilTempNl
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

c = 86400 # seconds in a day
CO2_PPM = 400 # assumed constant outdoor co2 concentration [ppm]
weatherDataDir = "environments/data/seljaar.mat"
mat = loadmat(weatherDataDir)

rawWeather = mat['seljaarhires']
time = rawWeather[:,0] # time in [s]
dt = np.mean(np.diff(time-time[0])) # sample period of data [s]

h = 3 # sample period for the solver [s]

startDay = 0
nDays = 1
predHorizon = 1 # [days]

Ns = int(np.ceil(nDays*c/dt))    # Number of samples we need
N0 = int(np.ceil(startDay*c/dt)) # Start index

Np = int(np.ceil(predHorizon*c/dt))
print(Np)

# p = int(np.floor(1/(dt/h))) # new sample period?
# print(p)

# p = (dt/h)

weatherData = np.zeros((Ns+Np, 7))
time = time[N0:N0+Ns+Np]
weatherData[:, 0] = rawWeather[N0:N0+Ns+Np, 1]          # iGlob
weatherData[:, 1] = rawWeather[N0:N0+Ns+Np, 3] + 1.5    # tOut

# convert relative humidity to vapor density
vpDensity = rh2vaporDens(weatherData[:, 1], rawWeather[N0:N0+Ns+Np, 8])     # vp Density

# convert vapor density to vapor pressure
weatherData[:,2] = vaporDens2pres(weatherData[N0:N0+Ns+Np, 1], vpDensity)   # vpOut
weatherData[:,3] = co2ppm2dens(weatherData[N0:N0+Ns+Np, 1], CO2_PPM)*1e6    # co2Out (converted from kg/m^3 to mg/m^3)
weatherData[:,4] = rawWeather[N0:N0+Ns+Np, 2]               # wind
weatherData[:,5] = rawWeather[N0:N0+Ns+Np, 4]               # tSky
weatherData[:,6] = soilTempNl(rawWeather[N0:N0+Ns+Np, 0])   # tSoOut

ns = int((dt/h) * (Ns+Np))

# weatherDataResampled = weatherData[0] + resample(weatherData-weatherData[0], ns-1)

interpolation = CubicSpline(time[N0:N0+Ns+Np], weatherData)
timeRes = np.linspace(time[0], time[-1], ns)
weatherDataResampled = interpolation(timeRes)

# for i in range():
#     plt.plot(time/c, weatherData[:, i])
#     plt.plot(timeRes, weatherDataResampled[:, i], "--")
#     plt.show()
# weatherDataResampled[0<]
# nDays = 1           # [days]

# Ns = int(np.ceil(nDays/dt))

# startDay/dt

# N0      = int(np.ceil(startDay/dt)) # Start index

# ns = Ns * 5/1

# iGlob = resample(weatherData[:Ns,0], Ns)

# # p       = int(np.floor(1/(dt/h)))
