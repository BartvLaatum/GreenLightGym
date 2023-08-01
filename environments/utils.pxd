from libc.math cimport exp

cdef inline double satVp(temp):
# saturated vapor pressure (Pa) at temperature temp (�C)
# Calculation based on 
#   http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
# See also file atmoclc2.pdf

    # parameters used in the conversion
    # p = [610.78 238.3 17.2694 -6140.4 273 28.916];
        # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]

        # Saturation vapor pressure of air in given temperature [Pa]
    return 610.78*exp(17.2694*temp/(temp+238.3))

cdef inline double cond(hec, vp1, vp2):
# COND Vapor flux from the air to an object by condensation in the Vanthoor model
# The vapor flux is measured in kg m^{-2} s^{-1}.
# Based on Equation 43 in the electronic appendix of 
#   Vanthoor, B., Stanghellini, C., van Henten, E. J. & de Visser, P. H. B. 
#       A methodology for model-based greenhouse design: Part 1, a greenhouse climate 
#       model for a broad range of designs and climates. Biosyst. Eng. 110, 363–377 (2011).
#
# Usage:
#   de = cond(hec, vp1, vp2)
#
# Inputs:
#   hec     the heat exchange coefficienct between object1 (air) and object2 (a surface) [W m^{-2} K^{-1}]
#   vp1     the vapor pressure of the air
#   vp2     the saturation vapor pressure at the temperature of the object
#
# Outputs:
#   de      a DynamicElement representing the condensation between object1 and object2

# David Katzin, Wageningen University
# david.katzin@wur.nl
# david.katzin1@gmail.com    

    # sMV12 = -0.1
    return 1/(1 + exp(-0.1*(vp1-vp2))) * 6.4e-9*hec*(vp1-vp2)

cdef inline double co2dens2ppm(temp, dens):
# CO2DENS2PPM Convert CO2 density [kg m^{-3}] to molar concetration [ppm] 
#
# Usage: 
#   ppm = co2dens2ppm(temp, dens)
# Inputs:
#   temp        given temperatures [�C] (numeric vector)
#   dens        CO2 density in air [kg m^{-3}] (numeric vector)
#   Inputs should have identical dimensions
# Outputs:
#   ppm         Molar concentration of CO2 in air [ppm] (numerical vector)
#
# calculation based on ideal gas law pV=nRT, pressure is assumed to be 1 atm

# David Katzin, Wageningen University
# david.katzin@wur.nl
# david.katzin1@gmail.com

    # R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    # C2K = 273.15 # conversion from Celsius to Kelvin [K]
    # M_CO2 = 44.01e-3 # molar mass of CO2 [kg mol^-{1}]
    # P = 101325 # pressure (assumed to be 1 atm) [Pa]
    return 10**6*8.3144598 * (temp+273.15) * dens/(101325*44.01e-3)
