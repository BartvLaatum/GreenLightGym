import scipy.io
import numpy as np

def loadWeatherData():
    """
        d[0]: iGlob         Global radiation [W m^{-2}]
        d[1]: tOut          Outdoor temperature [deg C]    
        d[2]: vpOut         Outdoor vapor pressure [Pa]
        d[3]: co2Out        Outdoor CO2 concentration [mg m^{-3}]
        d[4]: wind          Outdoor wind speed [m s^{-1}]
        d[5]: tSky          Sky temperature [deg C]
        d[6]: tSoOut        Outdoor soil temperature [deg C]
        d[7]: radSum        Daily global radiation [J m^{-2} day^{-1}]
    """
    CO2_PPM = 400
    mat = scipy.io.loadmat('environments/data/seljaar.mat')
    rawWeather = mat['seljaarhires']
    # rawWeather data is a file with 9 columns

    weatherData = np.zeros((len(rawWeather), 7))
    weatherData[:,0] = rawWeather[:,1] # iGlob
    weatherData[:,1] = rawWeather[:,3] +1.5 # tOut

    # convert relative humidity to vapor density
    vpDensity = rh2vaporDens(weatherData[:,1], rawWeather[:, 8]) # vpOut

    # convert vapor density to vapor pressure
    weatherData[:,2] = vaporDens2pres(weatherData[:,1], vpDensity) # vpOut
    weatherData[:,3] = co2ppm2dens(weatherData[:,1], CO2_PPM)*1e6 # co2Out (converted from kg/m^3 to mg/m^3)
    weatherData[:,4] = rawWeather[:, 2] # wind
    weatherData[:,5] = rawWeather[:, 4] # tSky
    weatherData[:,6] = soilTempNl(rawWeather[:, 0]) # tSoOut
    return weatherData

def soilTempNl(time):
#SOILTEMPNL An estimate of the soil temperature in the Netherlands in a given time of year
# Based on Figure 3 in 
#   Jacobs, A. F. G., Heusinkveld, B. G. & Holtslag, A. A. M. 
#   Long-term record and analysis of soil temperatures and soil heat fluxes in 
#   a grassland area, The Netherlands. Agric. For. Meteorol. 151, 774�780 (2011).
#
# Input:
#   time - seconds since beginning of the year [s]
# Output:
#   soilT - soil temperature at 1 meter depth at given time [�C]
#
# Calculated based on a sin function approximating the figure in the reference

# David Katzin, Wageningen University
# david.katzin@wur.nl

    SECS_IN_YEAR = 3600*24*365
    soilT = 10+5*np.sin((2*np.pi*(time+0.625*SECS_IN_YEAR)/SECS_IN_YEAR))
    return soilT

def vaporDens2pres(temp, vaporDens):
# VAPORDENS2PRES Convert vapor density [kg{H2O} m^{-3}] to vapor pressure [Pa]
#
# Usage:
#   vaporPres = vaporDens2pres(temp, vaporDens)
# Inputs:
#   temp        given temperatures [°C] (numeric vector)
#   vaporDens   vapor density [kg{H2O} m^{-3}] (numeric vector)
#   Inputs should have identical dimensions
# Outputs:
#   vaporPres   vapor pressure [Pa] (numeric vector)
#
# Calculation based on 
#   http://www.conservationphysics.org/atmcalc/atmoclc2.pdf

# David Katzin, Wageningen University
# david.katzin@wur.nl
# david.katzin1@gmail.com
    
    # parameters used in the conversion
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
        # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]
    
    rh = vaporDens/rh2vaporDens(temp, 100) # relative humidity [0-1]
        
    satP = p[0]*np.exp(p[2]*temp/(temp+p[1]))
        # Saturation vapor pressure of air in given temperature [Pa]
    
    return satP*rh


def co2ppm2dens(temp, ppm):
    # CO2PPM2DENS Convert CO2 molar concetration [ppm] to density [kg m^{-3}]

    # Usage:
    #   co2Dens = co2ppm2dens(temp, ppm) 
    # Inputs:
    #   temp        given temperatures [�C] (numeric vector)
    #   ppm         CO2 concetration in air (ppm) (numeric vector)
    #   Inputs should have identical dimensions
    # Outputs:
    #   co2Dens     CO2 concentration in air [kg m^{-3}] (numeric vector)

    # Calculation based on ideal gas law pV=nRT, with pressure at 1 atm

    # David Katzin, Wageningen University
    # david.katzin@wur.nl
    # david.katzin1@gmail.com

    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3 # molar mass of CO2 [kg mol^-{1}]
    P = 101325 # pressure (assumed to be 1 atm) [Pa]
    
    # number of moles n=m/M_CO2 where m is the mass [kg] and M_CO2 is the
    # molar mass [kg mol^{-1}]. So m=p*V*M_CO2*P/RT where V is 10^-6*ppm    
    return P*10**-6*ppm*M_CO2/(R*(temp+C2K))

def rh2vaporDens(temp, rh):
    # RH2VAPORDENS Convert relative humidity [#] to vapor density [kg{H2O} m^{-3}]

    # Usage:
    #   vaporDens = rh2vaporDens(temp, rh)
    # Inputs:
    #   temp        given temperatures [�C] (numeric vector)
    #   rh          relative humidity [#] between 0 and 100 (numeric vector)
    #   Inputs should have identical dimensions
    # Outputs:
    #   vaporDens   absolute humidity [kg{H20} m^{-3}] (numeric vector)

    # Calculation based on 
    #   http://www.conservationphysics.org/atmcalc/atmoclc2.pdf

    # David Katzin, Wageningen University
    # david.katzin@wur.nl
    # david.katzin1@gmail.com

    # constants
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    Mw = 18.01528e-3 # molar mass of water [kg mol^-{1}]
    
    # parameters used in the conversion
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]
    
    satP = p[0]*np.exp(p[2]*temp/(temp+p[1]))
        # Saturation vapor pressure of air in given temperature [Pa]
    
    pascals=(rh/100)*satP # Partial pressure of vapor in air [Pa]
    
    # convert to density using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.
    
    return pascals*Mw/(R*(temp+C2K))

