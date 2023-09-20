"""
Create a cython class that contains the GreenLight model data structure and functions.
This class will be used to create a cython module of GreenLight that can be imported into python environment.

The python environment will send setpoints as actions to the cython module.
Next, cython will compute the control signals, and simulate the new state of the greenhouse.
Finally, the new state/measurement/disturbances will be returned to the python environment.
"""
from auxiliaryStates cimport AuxiliaryStates, update
from defineParameters cimport Parameters, initParameters
from differenceFunction cimport fRK4
from computeControls cimport controlSignal
from utils cimport satVp, co2dens2ppm
from libc.stdlib cimport malloc, free
import cython

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef class GreenLight:
    cdef Parameters* p      # pointer to Parameters struct
    cdef AuxiliaryStates* a # pointer to AuxiliaryStates struct
    cdef double (*d)[10]    # pointer to weather data
    cdef double* x          # pointer to states
    cdef double* u          # pointer to control signals
    cdef float h            # step size
    cdef unsigned int timestep  # current timestep
    cdef char nx            # number of states
    cdef char nu            # number of control signals
    cdef char nd            # number of disturbances
    cdef unsigned short solverSteps # number of steps to take by solver between time interval for observing the env

    def __cinit__(self,
                cnp.ndarray[cnp.double_t, ndim=2] weather,
                float h,
                char nx,
                char nu,
                char nd,
                char noLamps,
                char ledLamps,
                char hpsLamps,
                char intLamps,
                unsigned short solverSteps,
                unsigned int timeInDays):

        self.p = <Parameters*>malloc(sizeof(Parameters))
        self.a = <AuxiliaryStates*>malloc(sizeof(AuxiliaryStates))
        self.u = <double*>malloc(nu * sizeof(double))
        initParameters(self.p, noLamps, ledLamps, hpsLamps, intLamps)
        self.h = h
        self.nx = nx
        self.nu = nu
        self.nd = nd
        self.timestep = 0
        self.solverSteps = solverSteps
        self.initWeather(weather)
        self.initStates(self.d[0], timeInDays)

        # compute auxiliary states once before start of simulation
        # update(self.a, self.p, self.u, self.x, self.d[0])

    def __dealloc__(self):
        free(self.a)
        free(self.p)
        free(self.d)
        free(self.x)
        free(self.u)

    def setTimestep(self, int timestep):
        self.timestep = timestep

    def reset(self, unsigned int timeInDays):
        self.timestep = 0
        self.initStates(self.d[0], timeInDays)

    def setNightCo2(self, short co2SpNight):
        self.p.co2SpNight = co2SpNight

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef void step(self, cnp.ndarray[cnp.float32_t, ndim=1] controls, cnp.ndarray[char, ndim=1] learnedControlIdx):
        """
        Simulate the state of the system at the next time step using the GreenLight model.
        Inputs are the controls signals that are computed in the python environment, e.g., by an RL-agent.
        The control signals are copied to a c-type array, such that they can be used in the GreenLight model.
        Args:
            controls (np.ndarray)           - Array with control signals.
            learnedControlIdx (np.ndarray)  - Array with indices of control signals that are learned.
        """
        cdef unsigned char i
        cdef unsigned short j

        # convert control numpy array control inputs into c-type array
        # we have controls inputs that are based on setpoits (computed here)
        # and we have control inputs that are learned (computed in python environment)
        # compute control signal at specific time step
        self.u = controlSignal(self.p, self.x, self.u, self.d[self.timestep*self.solverSteps])

        for i in range(len(learnedControlIdx)):
            self.u[learnedControlIdx[i]] = controls[i]

        self.a.mcFruitHarSum = 0
        for j in range(self.solverSteps):
            self.x = fRK4(self.a, self.p, self.u, self.x, self.d[self.timestep * self.solverSteps + j], self.h, self.nx)
        self.timestep += 1

    cdef void initWeather(self, cnp.ndarray[cnp.double_t, ndim=2] weather):
        """
        Function to initialize the weather data in the cython module of GreenLight.
        We copy the array with weather data that was loaded in by the python environment to an array in the cython module.

        Args:
            weather (np.ndarray): Array with weather data.
        """
        cdef int i, j
        cdef cnp.ndarray[cnp.double_t, ndim=2] np_weather = np.asarray(weather, dtype=np.double)
        cdef int n = np_weather.shape[0]
        cdef char l = self.nd
        self.d = <double(*)[10]>malloc(n * sizeof(double[10]))

        for i in range(n):
            for j in range(l):
                self.d[i][j] = np_weather[i, j]

    cpdef void setStates(self, cnp.ndarray[cnp.double_t, ndim=1] states):
        """
        Function to set the states of the GreenLight model.
        Except for the state that we compute using the model.
        Usefull for testing.
        We copy the array with states that was loaded in by the python environment to an array in the cython module.

        Args:
            states (np.ndarray): Array with states.
        """
        cdef unsigned char i
        cdef cnp.ndarray[cnp.double_t, ndim=1] np_states = np.asarray(states, dtype=np.double)
        # cdef int n = np_states.shape[0]
        for i in range(self.nx):
            # if i != testIndex:
            self.x[i] = np_states[i]

    cdef void initStates(self, double* d0, unsigned int timeInDays):
        """
        CO2 concentration is equal to outdoor CO2	
        x[0]: co2Air    CO2 concentration in main air compartment [mg m^{-3}]
        x[1]: co2Top    CO2 concentration in top air compartment [mg m^{-3}]
        x[2]: tAir      Air temperature in main compartment [deg C]
        x[3]: tTop      Air temperature in top compartment [deg C]
        x[4]: tCan      Temperature of the canopy [deg C]
        x[5]: tCovIn    Indoor cover temperature [deg C]
        x[6]: tCovE     Outdoor cover temperature [deg C]
        x[7]: tThScr    Thermal screen temperature [deg C]
        x[8]: tFlr      Floor temperature [deg C]
        x[9]: tPipe     Pipe temperature [deg C]
        x[10]: tSoil1   First soil layer temperature [deg C]
        x[11]: tSoil2   Second soil layer temperature [deg C]
        x[12]: tSoil3   Third soil layer temperature [deg C]
        x[13]: tSoil4   Fourth soil layer temperature [deg C]
        x[14]: tSoil5   Fifth soil layer temperature [deg C]
        x[15]: vpAir    Vapor pressure of main air compartment [Pa]
        x[16]: vpTop    Vapor pressure of top air compartment [Pa]
        x[17]: tLamp    Lamp temperature [deg C]
        x[18]: tIntLamp Interlight temperature [deg C]
        x[19]: tGroPipe Grow pipe temperature [deg C]
        x[20]: tBlScr   Blackout screen temperature [deg C]
        x[21]: tCan24   Average temperature of the canopy over last 24 hours [deg C]
        
        x[22]: cBuf     Carbohydrates in crop buffer [mg{CH20} m^{-2}]
        x[23]: cLeaf    Carbohydrates in leaves [mg{CH20} m^{-2}]
        x[24]: cStem    Carbohydrates in stem [mg{CH20} m^{-2}]
        x[25]: cFruit   Carbohydrates in fruit [mg{CH20} m^{-2}]
        x[26]: tCanSum  Crop development stage [C day]

        x[27]: time     Time since 00-00-0000 [days]
    """
        # self.x = <double(*)[26]>malloc(sizeof(double))
        # Air and vapor pressure are assumed to start at the night setpoints
        # x.co2Air.val = d.co2Out.val(1,2)
        self.x = <double*>malloc(self.nx * sizeof(double))
        self.x[0] = d0[3] # co2Air

        # x.co2Top.val = x.co2Air.val
        self.x[1] = self.x[0] # co2Top

        # x.tAir.val = p.tSpNight.val
        self.x[2] = self.p.tSpNight # tAir

        # x.tTop.val = x.tAir.val
        self.x[3] = self.x[2] # tTop

        # x.tCan.val = x.tAir.val+4
        self.x[4] = self.x[2] + 4

        # x.tCovIn.val = x.tAir.val
        self.x[5] = self.x[2]

        # x.tCovE.val = x.tAir.val
        self.x[6] = self.x[2]

        # x.tThScr.val = x.tAir.val
        self.x[7] = self.x[2]

        # x.tFlr.val = x.tAir.val
        self.x[8] = self.x[2]

        # x.tPipe.val = x.tAir.val
        self.x[9] = self.x[2]

        # x.tSo1.val = x.tAir.val
        self.x[10] = self.x[2]

        # x.tSo2.val = 1/4*(3*x.tAir.val+d.tSoOut.val(1,2))
        self.x[11] = 1/4*(3*self.x[2] + d0[6])

        # # x.tSo3.val = 1/4*(2*x.tAir.val + 2*d.tSoOut.val(1,2))
        self.x[12] = 1/4*(2*self.x[2] + 2*d0[6])

        # # x.tSo4.val = 1/4*(x.tAir.val+3*d.tSoOut.val(1,2))
        self.x[13] = 1/4*(self.x[2] + 3*d0[6])

        # # x.tSo5.val = d.tSoOut.val(1,2)
        self.x[14] = d0[6]

        # # x.vpAir.val = p.rhMax.val/100*satVp(x.tAir.val)
        self.x[15] = self.p.rhMax/100*satVp(self.x[2])

        # # x.vpTop.val = x.vpAir.val
        self.x[16] = self.x[15]

        # # x.tLamp.val = x.tAir.val
        self.x[17] = self.x[2]

        # # x.tIntLamp.val = x.tAir.val
        self.x[18] = self.x[2]

        # # x.tGroPipe.val = x.tAir.val
        self.x[19] = self.x[2]

        # # x.tBlScr.val = x.tAir.val
        self.x[20] = self.x[2]
        
        # x.tCan24.val = x.tCan.val
        self.x[21] = self.x[4]

        ## crop model
        # x.cBuf.val = 0
        self.x[22] = 0

        # # start with 3.12 plants/m2, assume they are each 2 g = 6240 mg/m2.
        self.x[23] = 0.7*6240   # 70% in leafs
        self.x[24] = 0.25*6240  # 25% in stems
        self.x[25] = 0.05*6240 # 5% in fruits we only harvest if this is > 300K

        # x.tCanSum.val = 0
        self.x[26] = 0

        # time since 01-01-0001 [days]:
        self.x[27] = timeInDays


    cpdef getWeatherArray(self):
        """
        Function that copies weather data from the cython module to a numpy array.
        Such that we can acces the weather data in the python environment.
        Currently copies complete array with weather data, but this can be changed to only copy the relevant data.
        For example, a future weather prediction.
        """
        cdef int n = 105121
        cdef unsigned char m = self.nd
        cdef  cnp.ndarray[cnp.double_t, ndim=1] np_d = np.zeros((m), dtype=np.double)
        for i in range(n):
            for j in range(m):
                np_d[i, j] = self.d[i][j]
        return np_d

    cpdef getControlsArray(self):
        """
        Function that copies the control signals from the cython module to a numpy array.
        Such that we can acces the control signals in the python environment.
        """
        cdef unsigned char i
        cdef  cnp.ndarray[cnp.double_t, ndim=1] np_u = np.zeros(self.nu, dtype=np.double)
        for i in range(self.nu):
            np_u[i] = self.u[i]
        return np_u

    cpdef getStatesArray(self):
        """
        Function that copies the states from the cython module to a numpy array.
        Such that we can acces the states in the python environment.
        """
        cdef unsigned char i
        cdef  cnp.ndarray[cnp.double_t, ndim=1] np_x = np.zeros(self.nx, dtype=np.double)
        for i in range(self.nx):
            np_x[i] = self.x[i]
        return np_x

    cpdef getObs(self):
        """
        Function that copies the observations from the cython module to a numpy array.
        Such that we can acces the observations in the python environment.
        """
        cdef  cnp.ndarray[cnp.double_t, ndim=1] np_obs = np.zeros(6, dtype=np.double)
        np_obs[0] = self.x[2]                               # Air temperature in main compartment [deg C]
        np_obs[1] = self.a.co2InPpm                         # CO2 concentration in main air compartment [ppm]
        np_obs[2] = 100*self.x[15]/satVp(self.x[2])         # Relative humidity in main air compartment [%]
        np_obs[3] = self.x[25]*1e-6                         # Fruit dry matter weight [kg{CH20} m^{-2}]
        np_obs[4] = self.a.mcFruitHarSum*1e-6               # Harvested fruit in dry matter weight [kg{CH20} m^{-2}]
        np_obs[5] = self.a.rParGhSun + self.a.rParGhLamp    # PAR radiation above the canopy [W m^{-2}]
        return np_obs

    @property
    def co2SpNight(self):
        return self.p.co2SpNight

    # CO2 injection rate [mg m^-2 s^-1]
    @property
    def co2InjectionRate(self):
        return self.a.mcExtAir

    # returns the number of states
    @property
    def nx(self):
        return self.nx

    @property
    # returns the number of control signals
    def nu(self):
        return self.nu

    @property
    # returns the number of disturbances
    def nd(self):
            return self.nd

    @property
    # returns amount of PAR radiation above the canopy from the sun
    def rParGhSun(self):
        return self.a.rParGhSun
    
    @property
    # returns the amount of PAR radiation above the canopy from the lamps
    def rParGhLamp(self):
        return self.a.rParGhLamp

    @property
    # returns the current timestep
    def timestep(self):
        return self.timestep

    @property
    # returns the co2 concentration in the air in [ppm]
    def co2InPpm(self):
        return self.a.co2InPpm

    @property
    def rhIn(self):
        return self.a.rhIn

    @property
    def time(self):
        return self.x[27]

    # property lampTimeOfDay:
    #     def __get__(self):
    #         return self.a.lampTimeOfDay