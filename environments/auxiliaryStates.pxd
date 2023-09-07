# Import the Parameters struct from defineParameters.pxd
from defineParameters cimport Parameters
from libc.math cimport cos, M_PI, exp, sqrt, fabs, fmax, fmin, log
from utils cimport satVp, cond, co2dens2ppm

cdef packed struct AuxiliaryStates:
    # """
    # Auxiliary states of  GreenLight model.
    # """

    #####################################################
    #### Shading Screen and Permanent shading screen ####
    ##################################################### 
    # double tauShScrPar
    # double tauShScrPerPar
    # double rhoShScrPar
    # double rhoShScrPerPar
    # double tauShScrShScrPerPar
    # double rhoShScrShScrPerParUp
    # double rhoShScrShScrPerParDn
    # double tauShScrNir
    # double tauShScrPerNir
    # double rhoShScrNir
    # double rhoShScrPerNir
    # double tauShScrShScrPerNir
    # double rhoShScrShScrPerNirUp
    # double rhoShScrShScrPerNirDn

    # double tauShScrFir
    # double tauShScrPerFir
    # double rhoShScrFir
    # double rhoShScrPerFir
    # double tauShScrShScrPerFir
    # double rhoShScrShScrPerFirUp
    # double rhoShScrShScrPerFirDn

    #################################
    #### Thermal Screen and Roof ####
    #################################
    double tauThScrPar
    double rhoThScrPar
    double tauCovThScrPar
    double rhoCovThScrParUp
    double rhoCovThScrParDn
    double tauThScrNir
    double rhoThScrNir
    double tauCovThScrNir
    double rhoCovThScrNirUp
    double rhoCovThScrNirDn

    #############################################
    #### all 4 layers of the Vanthoor model #####
    #############################################
    # double tauCovParOld
    # double rhoCovParOldUp
    # double rhoCovParOldDn
    # double tauCovNirOld
    # double rhoCovNirOldUp
    # double rhoCovNirOldDn

    double tauBlScrPar
    double rhoBlScrPar
    double tauCovBlScrPar
    double rhoCovBlScrParUp
    double rhoCovBlScrParDn
    double tauBlScrNir
    double rhoBlScrNir
    double tauCovBlScrNir
    double rhoCovBlScrNirUp
    double rhoCovBlScrNirDn

    ###################################
    #### All layers of GL model    ####
    ###################################

    double tauCovPar
    double rhoCovPar
    double tauCovNir
    double rhoCovNir
    double tauCovFir
    double rhoCovFir
    double aCovPar
    double aCovNir
    double aCovFir
    double epsCovFir
    double capCov

    ####################################
    #### Capacities - Section 4 [1] ####
    ####################################
    double lai
    double capCan
    double capCovE
    double capCovIn
    double capVpAir
    double capVpTop

    ############################################################
    #### Global, PAR, and NIR heat fluxes - Section 5.1 [1] ####
    ############################################################

    double qLampIn
    double qIntLampIn
    double rParGhSun
    double rParGhLamp
    double rParGhIntLamp
    double rCanSun
    double rCanLamp
    double rCanIntLamp
    double rCan
    double rParSunCanDown
    double rParLampCanDown
    double fIntLampCanPar

    double fIntLampCanNir
    double rParIntLampCanDown
    double rParSunFlrCanUp
    double rParLampFlrCanUp
    double rParIntLampFlrCanUp
    double rParSunCan
    double rParLampCan
    double rParIntLampCan
    double tauHatCovNir
    double tauHatFlrNir
    double tauHatCanNir
    double rhoHatCanNir

    double tauCovCanNir
    double rhoCovCanNirUp
    double rhoCovCanNirDn
    double tauCovCanFlrNir
    double rhoCovCanFlrNir
    double aCanNir
    double aFlrNir
    double rNirSunCan
    double rNirLampCan
    double rNirIntLampCan
    double rNirSunFlr
    double rNirLampFlr

    double rNirIntLampFlr
    double rParSunFlr
    double rParLampFlr
    double rParIntLampFlr
    double rLampAir
    double rIntLampAir
    double rGlobSunAir
    double rGlobSunCovE

    ###########################################
    #### FIR heat fluxes - Section 5.2 [1] ####
    ###########################################

    double tauThScrFirU
    double tauBlScrFirU
    double aCan
    double rCanCovIn
    double rCanSky
    double rCanThScr
    double rCanFlr
    double rPipeCovIn
    double rPipeSky
    double rPipeThScr
    double rPipeFlr
    double rPipeCan
    double rFlrCovIn
    double rFlrSky
    double rFlrThScr
    double rThScrCovIn
    double rThScrSky
    double rCovESky
    double rFirLampFlr
    double rLampPipe
    double rFirLampCan
    double rLampThScr
    double rLampCovIn
    double rLampSky
    double rGroPipeCan
    double rFlrBlScr
    double rPipeBlScr
    double rCanBlScr
    double rBlScrThScr
    double rBlScrCovIn
    double rBlScrSky
    double rLampBlScr
    double fIntLampCanUp
    double fIntLampCanDown
    double rFirIntLampFlr
    double rIntLampPipe
    double rFirIntLampCan
    double rIntLampLamp
    double rIntLampBlScr
    double rIntLampThScr
    double rIntLampCovIn
    double rIntLampSky    

    ###########################################
    #### Natural Ventilation - Section 9.7 ####
    ###########################################
    double aRoofU
    double aRoofUMax
    double aRoofMin
    double aSideU
    double etaRoof
    double etaRoofNoSide
    double etaSide
    double cD
    double cW
    double fVentRoof2
    double fVentRoof2Max
    double fVentRoof2Min
    double fVentRoofSide2
    double fVentSide2
    double fLeakage
    double fVentRoof
    double fVentSide

    #######################
    #### Control rules ####
    #######################
    double co2InPpm
    double rhIn

    ###################################
    #### Convection and Conduction ####
    ###################################

    double rhoTop
    double rhoAir
    double rhoAirMean
    double fThScr
    double fBlScr
    double fScr

    ###############################################
    #### Convective and conductive heat fluxes ####
    ###############################################

    char fVentForced
    double hCanAir
    double hAirFlr
    double hAirThScr
    double hAirBlScr
    double hAirOut
    double hAirTop
    double hThScrTop
    double hBlScrTop
    double hTopCovIn
    double hTopOut
    double hCovEOut
    double hPipeAir
    double hFlrSo1
    double hSo1So2
    double hSo2So3
    double hSo3So4
    double hSo4So5
    double hSo5SoOut
    double hCovInCovE
    double hLampAir
    double hGroPipeAir
    double hIntLampAir

    ##############################
    #### Canopy transpiration ####
    ##############################

    double sRs
    double cEvap3
    double cEvap4
    double rfRCan
    double rfCo2
    double rfVp
    double rS
    double vecCanAir
    double mvCanAir

    #######################
    #### Vapor Fluxes #####
    #######################

    char mvPadAir
    char mvFogAir
    char mvBlowAir
    char mvAirOutPad

    double mvAirThScr
    double mvAirBlScr
    double mvTopCovIn
    double mvAirTop
    double mvTopOut
    double mvAirOut

    ############################
    #### Latent heat fluxes ####
    ############################

    double lCanAir
    double lAirThScr
    double lAirBlScr
    double lTopCovIn

    ###############################
    #### Canopy photosynthesis ####
    ###############################

    double parCan
    double j25CanMax
    double gamma
    double co2Stom
    double jPot
    double j
    double p
    double r
    double hAirBuf
    double mcAirBuf
    double gTCan24
    double hTCan24
    double hTCan
    double hTCanSum
    double hBufOrg
    double mcBufLeaf
    double mcBufStem
    double mcBufFruit

    ############################################
    #### Growth and maintenance respiration ####
    ############################################
    double mcBufAir
    double mcLeafAir
    double mcStemAir
    double mcFruitAir
    double mcOrgAir
    double mcLeafHar
    double mcFruitHar
    double mcFruitHarSum

    ####################
    #### Co2 Fluxes ####
    ####################

    double mcAirCan
    double mcAirTop
    double mcTopOut
    double mcAirOut
    double hBoilPipe
    double hBoilGroPipe
    double mcExtAir

    # Lamp Cooling
    double hLampCool

    ##########################
    #### Currently unused ####
    ##########################
    char mcBlowAir
    char mcPadAir
    char hPadAir
    char hPasAir
    char hBlowAir
    char hAirPadOut
    char hAirOutPad
    char lAirFog
    char hIndPipe
    char hGeoPipe

    char hecMechAir
    char hAirMech
    char mvAirMech
    char lAirMech
    char hBufHotPipe

cdef inline double tau12(double tau1, double tau2, double rho1Dn, double rho2Up):
    """
    Transmission coefficient of a double layer [-]
    Equation 14 [1], Equation A4 [5]
    """
    return tau1*tau2/(1-rho1Dn*rho2Up)

cdef inline double rhoUp(double tau1, double rho1Up, double rho1Dn, double rho2Up):
    """
    Reflection coefficient of the upper layer [-]
    Equation 15 [1], Equation A5 [5]
    """
    return rho1Up + (tau1**2 *rho2Up)/(1-rho1Dn*rho2Up)

cdef inline double rhoDn(double tau2, double rho1Dn, double rho2Up, double rho2Dn):
    """
    Reflection coefficient of the upper layer [-]
    Equation 15 [1], Equation A5 [5]
    """
    return rho2Dn + (tau2**2*rho1Dn)/(1-rho1Dn*rho2Up)

cdef inline double radToDegrees(double degrees):
    return degrees*M_PI / 180.0

cdef inline double fir(double a1, double eps1, double eps2, double f12, double t1, double t2, double sigma):
    # Net far infrared flux from 1 to 2 [W m^{-2}]
    # Equation 37 [1]
    
    # sigma = 5.67e-8 we have this one in the defineParameters.pxd file
    # kelvin = 273.15

    return a1 * eps1 * eps2 * f12 * sigma * ((t1+273.15)**4 - (t2+273.15)**4)

cdef inline double sensible(double hec, double t1, double t2):
    # Sensible heat flux from 1 to 2 [W m^{-2}]
    # Equation 38 [1]
    return fabs(hec) * (t1 - t2)

cdef inline double airMv(double f12, double vp1, double vp2, double t1, double t2):
# Vapor flux accompanying an air flux [kg m^{-2} s^{-1}]
# Equation 44 [1]
    # mWater = 18
    # r = 8.314e3
	# kelvin = 273.15

 return (18/8.314e3)*fabs(f12) * (vp1/(t1+273.15) - vp2/(t2+273.15))

cdef inline double smoothHar(double processVar, double cutOff, double smooth, double maxRate):
    # Define a smooth function for harvesting (leaves, fruit, etc)
    # processVar - the DynamicElement to be controlled
    # cutoff     - the value at which the processVar should be harvested
    # smooth     - smoothing factor. The rate will go from 0 to max at
    #              a range with approximately this width
    # maxRate    - the maximum harvest rate
    return maxRate / (1 + exp(-(processVar-cutOff)*2 * log(100)/smooth))

cdef inline double airMc(double f12, double c1, double c2):
    # Co2 flux accompanying an air flux [kg m^{-2} s^{-1}]
    # Equation 45 [1]
    return fabs(f12)*(c1-c2)

# Function to update the auxiliary states based on the Parameters struct
cdef inline void update(AuxiliaryStates* a, Parameters* p, double &u[11], double &x[27], double &d[7]):
    """
    Update the auxiliary states based on the Parameters struct and previous auxiliary states.

    Args:
        p: Parameters struct
        u: array of control inputs
        a: AuxiliaryStates struct

        In u the following control inputs are expected:
        u[0]: boil          boiler valve (1 is full capacity)
        u[1]: extCo2        external CO2 valve (1 is full capacity)
        u[2]: thScr         closure of thermal screen (0 is open 1 is closed)
        u[3]: roof          roof vent aperature (0 is closed 1 is open)
        u[4]: lamps         artificial lighting (0 is off 1 is on)
        u[5]: intLamp       artificial interlights (0 is off 1 is on)
        u[6]: boilGro       boiler grow pipes valves (1 is full capacity)
        u[7]: blScr        closure blackout screen (0 is open 1 is closed)
        u[8]: shScr         closure of shading screen (0 is open 1 is closed)
        u[9]: shScrPer      closure of semi-permanent shading screen (0 is open 1 is closed)
        u[10]: side          side vent aperature (0 is closed 1 is open)

        In x the following greenhouse states are expected:
        x[0]: co2Air        CO2 concentration in main air compartment [mg m^{-3}]
        x[1]: co2Top        CO2 concentration in top air compartment [mg m^{-3}]
        x[2]: tAir          Air temperature in main compartment [deg C]
        x[3]: tTop          Air temperature in top compartment [deg C]
        x[4]: tCan          Temperature of the canopy [deg C]
        x[5]: tCovIn        Indoor cover temperature [deg C]
        x[6]: tCovE         Outdoor cover temperature [deg C]
        x[7]: tThScr        Thermal screen temperature [deg C]
        x[8]: Flr           Floor temperature [deg C]
        x[9]: tPipe         Pipe temperature [deg C]
        x[10]: tSoil1       First soil layer temperature [deg C]
        x[11]: tSoil2       Second soil layer temperature [deg C]
        x[12]: tSoil3       Third soil layer temperature [deg C]
        x[13]: tSoil4       Fourth soil layer temperature [deg C]
        x[14]: tSoil5       Fifth soil layer temperature [deg C]
        x[15]: vpAir        Vapor pressure of main air compartment [Pa]
        x[16]: vpTop        Vapor pressure of top air compartment [Pa]
        x[17]: tLamp        Lamp temperature [deg C]
        x[18]: tIntLamp     Interlight temperature [deg C]
        x[19]: tGroPipe     Grow pipe temperature [deg C]
        x[20]: tBlScr       Blackout screen temperature [deg C]
        x[21]: tCan24       Average of the canopy last 24 hours [deg C]

        x[22]: cBuf         Carbohydrates in crop buffer [mg{CH20} m^{-2}]
        x[23]: cLeaf        Carbohydrates in leaves [mg{CH20} m^{-2}]
        x[24]: cStem        Carbohydrates in stem [mg{CH20} m^{-2}]
        x[25]: cFruit       Carbohydrates in fruit [mg{CH20} m^{-2}]
        x[26]: tCanSum      Crop development stage [C day]

        In d the following weather disturbances are expected:
        d[0]: iGlob         Global radiation [W m^{-2}]
        d[1]: tOut          Outdoor temperature [deg C]    
        d[2]: vpOut         Outdoor vapor pressure [Pa]
        d[3]: co2Out        Outdoor CO2 concentration [mg m^{-3}]
        d[4]: wind          Outdoor wind speed [m s^{-1}]
        d[5]: tSky          Sky temperature [deg C]
        d[6]: tSoOut        Outdoor soil temperature [deg C]
    """

    #####################################################
    #### Shading Screen and Permanent shading screen ####
    ##################################################### 

    # PAR transmission coefficient of the shadow screen layer [-]
    # a.tauShScrPar = 1#1-u[8]*(1-p.tauShScrPar)

    # PAR transmission coefficient of the semi-permanent shadow screen layer [-]
    # a.tauShScrPerPar = 1 #  1-u[9]*(1-p.tauShScrPerPar)

    # PAR reflection coefficient of the shadow screen layer [-]
    # a.rhoShScrPar = 0# u[8] * p.rhoShScrPar

    # PAR reflection coefficient of the semi-permanent shadow screen layer [-]
    # a.rhoShScrPerPar = 0# u[9] * p.rhoShScrPerPar

    # PAR transmission coefficient of the shadow screen and semi permanent shadow screen layer [-]
    # Equation 16 [1]
    # a.tauShScrShScrPerPar = 1# tau12(a.tauShScrPar, a.tauShScrPar, a.rhoShScrPar, a.rhoShScrPar)

    # PAR reflection coefficient of the shadow screen and semi permanent shadow screen layer towards the top [-]
    # Equation 17 [1]
    # a.rhoShScrShScrPerParUp = 0#rhoUp(a.tauShScrPar, a.rhoShScrPar, a.rhoShScrPar, a.rhoShScrPar)

    # PAR reflection coefficient of the shadow screen and semi permanent shadow screen layer towards the bottom [-]
    # Equation 17 [1]
    # a.rhoShScrShScrPerParDn = 0#rhoDn(a.tauShScrPar, a.rhoShScrPar, a.rhoShScrPar, a.rhoShScrPar)

    # NIR transmission coefficient of the shadow screen layer [-]
    # a.tauShScrNir = 1 #1-u[8]*(1-p.tauShScrNir)

    # NIR transmission coefficient of the semi-permanent shadow screen layer [-]
    # a.tauShScrPerNir = 1 # 1-u[9]*(1-p.tauShScrPerNir)

    # NIR reflection coefficient of the shadow screen layer [-]
    # a.rhoShScrNir = 0 # u[8]*p.rhoShScrNir

    # NIR reflection coefficient of the semi-permanent shadow screen layer [-]
    # a.rhoShScrPerNir = 0 # u[9]*p.rhoShScrPerNir

    # NIR transmission coefficient of the shadow screen and semi permanent shadow screen layer [-]
    # a.tauShScrShScrPerNir = 1 # tau12(a.tauShScrNir, a.tauShScrPerNir, a.rhoShScrNir, a.rhoShScrPerNir)

    # NIR reflection coefficient of the shadow screen and semi permanent shadow screen layer towards the top [-]
    # a.rhoShScrShScrPerNirUp = 0 # rhoUp(a.tauShScrNir, a.rhoShScrNir, a.rhoShScrNir, a.rhoShScrPerNir)

    # NIR reflection coefficient of the shadow screen and semi permanent shadow screen layer towards the bottom [-]
    # a.rhoShScrShScrPerNirDn = 0 # rhoDn(a.tauShScrPerNir, a.rhoShScrNir, a.rhoShScrPerNir, a.rhoShScrPerNir)

    # FIR  transmission coefficient of the shadow screen layer [-]
    # a.tauShScrFir = 1 # 1-u[8]*(1-p.tauShScrFir)

    # # FIR transmission coefficient of the semi-permanent shadow screen layer [-]
    # a.tauShScrPerFir = 1 #1-u[9]*(1-p.tauShScrPerFir)

    # # FIR reflection coefficient of the shadow screen layer [-]
    # a.rhoShScrFir = 0 #u[8]*p.rhoShScrFir
    
    # # FIR reflection coefficient of the semi-permanent shadow screen layer [-]
    # a.rhoShScrPerFir = 0 # u[9]*p.rhoShScrPerFir
        
    # FIR transmission coefficient of the shadow screen and semi permanent shadow screen layer [-]
    # a.tauShScrShScrPerFir =  1 #tau12(a.tauShScrFir, a.tauShScrPerFir, a.rhoShScrFir, a.rhoShScrPerFir)
    
    # FIR reflection coefficient of the shadow screen and semi permanent shadow screen layer towards the top [-]
    # a.rhoShScrShScrPerFirUp = 0 # rhoUp(a.tauShScrFir, a.rhoShScrFir, a.rhoShScrFir, a.rhoShScrPerFir)
    
    # FIR reflection coefficient of the shadow screen and semi permanent shadow screen layer towards the bottom [-]
    # a.rhoShScrShScrPerFirDn = 0 # rhoDn(a.tauShScrPerFir, a.rhoShScrFir, a.rhoShScrPerFir, a.rhoShScrPerFir)

    #################################
    #### Thermal Screen and Roof ####
    #### PAR                	 ####
    #################################

    # PAR transmission coefficient of the thermal screen [-]
    a.tauThScrPar = 1-u[2]*(1-p.tauThScrPar)

    # PAR reflection coefficient of the thermal screen [-]
    a.rhoThScrPar = u[2]*(p.rhoThScrPar)
    # PAR transmission coefficient of the thermal screen and roof [-]
    a.tauCovThScrPar = tau12(p.tauRfPar, a.tauThScrPar, p.rhoRfPar, a.rhoThScrPar)

    # PAR reflection coefficient of the thermal screen and roof towards the top [-]
    a.rhoCovThScrParUp = rhoUp(p.tauRfPar, p.rhoRfPar, p.rhoRfPar, a.rhoThScrPar)

    # PAR reflection coefficient of the thermal screen and roof towards the bottom [-]
    a.rhoCovThScrParDn = rhoDn(a.tauThScrPar, p.rhoRfPar, a.rhoThScrPar, a.rhoThScrPar)

    #################################
    #### Thermal Screen and Roof ####
    #### NIR                	 ####
    #################################

    # NIR transmission coefficient of the thermal screen [-]
    a.tauThScrNir = 1-u[2]*(1-p.tauThScrNir)

    # NIR reflection coefficient of the thermal screen [-]
    a.rhoThScrNir = u[2]*p.rhoThScrNir

    # NIR transmission coefficient of the thermal screen and roof [-]
    a.tauCovThScrNir = tau12(p.tauRfNir, a.tauThScrNir, p.rhoRfNir, a.rhoThScrNir)

    # NIR reflection coefficient of the thermal screen and roof towards the top [-]
    a.rhoCovThScrNirUp = rhoUp(p.tauRfNir, p.rhoRfNir, p.rhoRfNir, a.rhoThScrNir)
    
    # NIR reflection coefficient of the thermal screen and roof towards the top [-]
    a.rhoCovThScrNirDn = rhoDn(a.tauThScrNir, p.rhoRfNir, a.rhoThScrNir, a.rhoThScrNir)

    #############################################
    #### all 4 layers of the Vanthoor model #####
    #############################################

    ## HERE THE REFLECTION AND TRANSMISSION OF THE THERMAL SCREEN AND THE SHADING SCREENS ARE COMBINED...
    ## IN PRACTICE THE VARIABLES WITH OLD ARE EXACTLY THE SAME AS THE THERMAL SCREENS...

    # Vanthoor PAR transmission coefficient of the cover [-]
    # a.tauCovThScrPar =  a.tauCovThScrPar #tau12(a.tauShScrShScrPerPar, a.tauCovThScrPar, a.rhoShScrShScrPerParDn, a.rhoCovThScrParUp)

    # Vanthoor PAR reflection coefficient of the cover towards the top [-]
    # a.rhoCovThScrParUp = a.rhoCovThScrParUp #rhoUp(a.tauShScrShScrPerPar, a.rhoShScrShScrPerParUp, a.rhoShScrShScrPerParDn, a.rhoCovThScrParUp)

    # Vanthoor PAR reflection coefficient of the cover towards the bottom [-]
    # a.rhoCovThScrParDn = a.rhoCovThScrParDn # rhoDn(a.tauCovThScrPar, a.rhoShScrShScrPerParDn, a.rhoCovThScrParUp, a.rhoCovThScrParDn)

    # Vanthoor NIR transmission coefficient of the cover [-]
    # a.tauCovNirOld = a.tauCovThScrNir #tau12(a.tauShScrShScrPerNir, a.tauCovThScrNir, a.rhoShScrShScrPerNirDn, a.rhoCovThScrNirUp)

    # Vanthoor NIR reflection coefficient of the cover towards the top [-]
    # a.rhoCovThScrNirUp = a.rhoCovThScrNirUp # rhoUp(a.tauShScrShScrPerNir, a.rhoShScrShScrPerNirUp, a.rhoShScrShScrPerNirDn, a.rhoCovThScrNirUp)

    # Vanthoor NIR reflection coefficient of the cover towards the bottom [-]
    # a.rhoCovThScrNirDn = a.rhoCovThScrNirDn #rhoDn(a.tauCovThScrNir, a.rhoShScrShScrPerNirDn, a.rhoCovThScrNirUp, a.rhoCovThScrNirDn)

    #############################################
    #### Vanthoor cover with blackout screen ####
    #############################################

    # PAR transmission coefficient of the blackout screen [-]
    a.tauBlScrPar = 1-u[7]*(1-p.tauBlScrPar)

    # PAR reflection coefficient of the blackout screen [-]
    a.rhoBlScrPar = u[7]*p.rhoBlScrPar

    # PAR transmission coefficient of the old cover and blackout screen [-]
	# Equation A9 [5]
    a.tauCovBlScrPar = tau12(a.tauCovThScrPar, a.tauBlScrPar, a.rhoCovThScrParDn, a.rhoBlScrPar)

    # PAR up reflection coefficient of the old cover and blackout screen [-]
	# Equation A10 [5]
    a.rhoCovBlScrParUp = rhoUp(a.tauCovThScrPar, a.rhoCovThScrParUp, a.rhoCovThScrParDn, a.rhoBlScrPar)
    
    # PAR down reflection coefficient of the old cover and blackout screen [-]
	# Equation A11 [5]
    a.rhoCovBlScrParDn = rhoDn(a.tauBlScrPar, a.rhoCovThScrParDn, a.rhoBlScrPar, a.rhoBlScrPar	)
    
    # NIR transmission coefficient of the blackout screen [-]
    a.tauBlScrNir = 1-u[7]*(1-p.tauBlScrNir)
    
    # NIR reflection coefficient of the blackout screen [-]
    a.rhoBlScrNir = u[7]*p.rhoBlScrNir
    
    # NIR transmission coefficient of the old cover and blackout screen [-]
    a.tauCovBlScrNir = tau12(a.tauCovThScrNir, a.tauBlScrNir, a.rhoCovThScrNirDn, a.rhoBlScrNir)

    # NIR up reflection coefficient of the old cover and blackout screen [-]
    a.rhoCovBlScrNirUp = rhoUp(a.tauCovThScrNir, a.rhoCovThScrNirUp, a.rhoCovThScrNirDn, a.rhoBlScrNir)
    
    # NIR down reflection coefficient of the old cover and blackout screen [-]
    a.rhoCovBlScrNirDn = rhoDn(a.tauBlScrNir, a.rhoCovThScrNirDn, a.rhoBlScrNir, a.rhoBlScrNir)

    ###################################
    #### All layers of GL model    ####
    ###################################

    # PAR transmission coefficient of the cover [-]
    # Equation A12 [5]
    a.tauCovPar = tau12(a.tauCovBlScrPar, p.tauLampPar, a.rhoCovBlScrParDn, p.rhoLampPar)

    # PAR reflection coefficient of the cover [-]
	# Equation A13 [5]
    a.rhoCovPar = rhoUp(a.tauCovBlScrPar, a.rhoCovBlScrParUp, a.rhoCovBlScrParDn, p.rhoLampPar)

    # NIR transmission coefficient of the cover [-]
    a.tauCovNir = tau12(a.tauCovBlScrNir, p.tauLampNir, a.rhoCovBlScrNirDn, p.rhoLampNir)

    # NIR reflection coefficient of the cover [-]
    a.rhoCovNir = rhoUp(a.tauCovBlScrNir, a.rhoCovBlScrNirUp, a.rhoCovBlScrNirDn, p.rhoLampNir)


    ## SINCE ONLY THE SHADING SCREEN AND THE ROOF HAVE AN EFFECT ON THE FIR TRANSMISSION AND REFLECTION
    ## WE CAN SIMPLY SET THIS TO THE FIR TRANSMISSION OF THE ROOF

    # FIR transmission coefficient of the cover, excluding screens and lamps [-]
    # a.tauCovFir = tau12(a.tauShScrShScrPerFir, p.tauRfFir, a.rhoShScrShScrPerFirDn, p.rhoRfFir)
    a.tauCovFir = p.tauRfFir

    # FIR reflection coefficient of the cover, excluding screens and lamps [-]
    # a.rhoCovFir = rhoUp(a.tauShScrShScrPerFir, a.rhoShScrShScrPerFirUp, a.rhoShScrShScrPerFirDn, p.rhoRfFir)
    a.rhoCovFir = p.rhoRfFir

    # PAR absorption coefficient of the cover [-]
    a.aCovPar = 1 - a.tauCovPar - a.rhoCovPar
    
    # NIR absorption coefficient of the cover [-]
    a.aCovNir = 1 - a.tauCovNir - a.rhoCovNir
    
    # FIR absorption coefficient of the cover [-]
    a.aCovFir = 1 - a.tauCovFir - a.rhoCovFir    

    # FIR emission coefficient of the cover [-]
    # See comment before equation 18 [1]
    a.epsCovFir = a.aCovFir

    # Heat capacity of the lumped cover [J K^{-1} m^{-2}]
    # Equation 18 [1]
    a.capCov = cos(radToDegrees(p.psi)) * (u[9] * p.hShScrPer * p.rhoShScrPer * p.cPShScrPer + p.hRf * p.rhoRf * p.cPRf)

    ####################################
    #### Capacities - Section 4 [1] ####
    ####################################

    # Leaf area index [m^2{leaf} m^{-2}]
    # Equation 5 [2]
    a.lai = p.sla * x[23]

    # Heat capacity of canopy [J K^{-1} m^{-2}]
    # Equation 19 [1]
    a.capCan = p.capLeaf * a.lai

    # Heat capacity of external and internal cover [J K^{-1} m^{-2}]
    # Equation 20 [1]
    a.capCovE = 0.1 * a.capCov
    a.capCovIn = 0.1 * a.capCov 

    # Vapor capacity of main compartment [kg m J^{-1}] 
    # Equation 24 [1]
    a.capVpAir = p.mWater * p.hAir / (p.R * (x[2] + 273.15))

    # Vapor capacity of top compartment [kg m J^{-1}] 
    a.capVpTop = p.mWater * (p.hGh - p.hAir) / (p.R * (x[3] + 273.15))

    ############################################################
    #### Global, PAR, and NIR heat fluxes - Section 5.1 [1] ####
    ############################################################

    # Lamp electrical input [W m^{-2}]
    # Equation A16 [5]
    a.qLampIn = p.thetaLampMax * u[4]

    # Interlight electrical input [W m^{-2}]
    # Equation A26 [5]
    a.qIntLampIn = p.thetaIntLampMax * u[5]

    # PAR above the canopy from the sun [W m^{-2}]
    # Equation 27 [1], Equation A14 [5]
    a.rParGhSun = (1 - p.etaGlobAir) * a.tauCovPar * p.etaGlobPar * d[0]

    # PAR above the canopy from the lamps [W m^{-2}] 
    # Equation A15 [5]
    a.rParGhLamp = p.etaLampPar * a.qLampIn
    
    # PAR outside the canopy from the interlights [W m^{-2}] 
    # Equation 7.7, 7.14 [7]
    a.rParGhIntLamp = p.etaIntLampPar * a.qIntLampIn
    
    # Global radiation above the canopy from the sun [W m^{-2}]
    # (PAR+NIR, where UV is counted together with NIR)
    # Equation 7.24 [7]
    a.rCanSun = (1 - p.etaGlobAir) * d[0] * (p.etaGlobPar * a.tauCovPar + p.etaGlobNir * a.tauCovNir)
    
    # Global radiation above the canopy from the lamps [W m^{-2}]
    # (PAR+NIR, where UV is counted together with NIR)
    # Equation 7.25 [7]
    a.rCanLamp = (p.etaLampPar + p.etaLampNir) * a.qLampIn

    # Global radiation outside the canopy from the interlight lamps [W m^{-2}]
    # (PAR+NIR, where UV is counted together with NIR)
    # Equation 7.26 [7]
    a.rCanIntLamp = (p.etaIntLampPar + p.etaIntLampNir) * a.qIntLampIn

    # Global radiation above and outside the canopy [W m^{-2}]
    # (PAR+NIR, where UV is counted together with NIR)
    # Equation 7.23 [7]
    a.rCan = a.rCanSun + a.rCanLamp + a.rCanIntLamp
    
    # PAR from the sun directly absorbed by the canopy [W m^{-2}]
    # Equation 26 [1]
    a.rParSunCanDown = a.rParGhSun * (1 - p.rhoCanPar) * (1 - exp(-p.k1Par * a.lai))
    
    # PAR from the lamps directly absorbed by the canopy [W m^{-2}]
    # Equation A17 [5]
    a.rParLampCanDown = a.rParGhLamp * (1 - p.rhoCanPar) * (1 - exp(-p.k1Par * a.lai))
    
    # Fraction of PAR from the interlights reaching the canopy [-]
    # Equation 7.13 [7]
    a.fIntLampCanPar = 1 - p.fIntLampDown * exp(-p.k1IntPar * p.vIntLampPos * a.lai) + \
    (p.fIntLampDown - 1) * exp(-p.k1IntPar * (1 - p.vIntLampPos) * a.lai)
        # Fraction going up and absorbed is (1-p.fIntLampDown)*(1-exp(-p.k1IntPar*(1-p.vIntLampPos)*gl.a.lai))
        # Fraction going down and absorbed is p.fIntLampDown*(1-exp(-p.k1IntPar*p.vIntLampPos*gl.a.lai))
        # This is their sum
        # e.g., if p.vIntLampPos==1, the lamp is above the canopy
        #   fraction going up and abosrbed is 0
        #   fraction going down and absroebd is p.fIntLampDown*(1-exp(-p.k1IntPar*gl.a.lai))

    # Fraction of NIR from the interlights reaching the canopy [-]
    # Analogous to Equation 7.13 [7]
    a.fIntLampCanNir = 1 - p.fIntLampDown * exp(-p.kIntNir * p.vIntLampPos * a.lai) + \
        (p.fIntLampDown - 1) * exp(-p.kIntNir * (1 - p.vIntLampPos) * a.lai)

    # PAR from the interlights directly absorbed by the canopy [W m^{-2}]
    # Equation 7.16 [7]
    a.rParIntLampCanDown = a.rParGhIntLamp * a.fIntLampCanPar * (1 - p.rhoCanPar)
    
    # PAR from the sun absorbed by the canopy after reflection from the floor [W m^{-2}]
    # Equation 28 [1]
    # addAux(gl, 'rParSunFlrCanUp', mulNoBracks(gl.a.rParGhSun, exp(-p.k1Par*gl.a.lai)*p.rhoFlrPar* \
    #     (1-p.rhoCanPar).*(1-exp(-p.k2Par*gl.a.lai))))
    a.rParSunFlrCanUp = a.rParGhSun * exp(-p.k1Par * a.lai) * p.rhoFlrPar * (1 - p.rhoCanPar) * (1 - exp(-p.k2Par * a.lai))

    # PAR from the lamps absorbed by the canopy after reflection from the floor [W m^{-2}]
    # Equation A18 [5]
    a.rParLampFlrCanUp = a.rParGhLamp * exp(-p.k1Par * a.lai) * p.rhoFlrPar * (1 - p.rhoCanPar) * (1 - exp(-p.k2Par * a.lai))

    # PAR from the interlights absorbed by the canopy after reflection from the floor [W m^{-2}]
    # Equation 7.18 [7]
    a.rParIntLampFlrCanUp = a.rParGhIntLamp * p.fIntLampDown *\
        exp(-p.k1IntPar * p.vIntLampPos * a.lai) * p.rhoFlrPar * \
        (1 - p.rhoCanPar) * (1 - exp(-p.k2IntPar * a.lai))
        # if p.vIntLampPos==1, the lamp is above the canopy, light loses
        # exp(-k*LAI) on its way to the floor.
        # if p.vIntLampPos==0, the lamp is below the canopy, no light is
        # lost on the way to the floor
    
    # Total PAR from the sun absorbed by the canopy [W m^{-2}]
    # Equation 25 [1]
    a.rParSunCan = a.rParSunCanDown + a.rParSunFlrCanUp
    
    # Total PAR from the lamps absorbed by the canopy [W m^{-2}]
    # Equation A19 [5]
    a.rParLampCan = a.rParLampCanDown + a.rParLampFlrCanUp

    # Total PAR from the interlights absorbed by the canopy [W m^{-2}]
    # Equation A19 [5], Equation 7.19 [7]
    a.rParIntLampCan = a.rParIntLampCanDown + a.rParIntLampFlrCanUp

    # Virtual NIR transmission for the cover-canopy-floor lumped model [-]
    # Equation 29 [1]
    a.tauHatCovNir = 1-a.rhoCovNir
    a.tauHatFlrNir = 1-p.rhoFlrNir

    # NIR transmission coefficient of the canopy [-]
    # Equation 30 [1]   
    a.tauHatCanNir = exp(-p.kNir * a.lai)
    
    # NIR reflection coefficient of the canopy [-]
    # Equation 31 [1]
    a.rhoHatCanNir = p.rhoCanNir * (1 - a.tauHatCanNir)

    # NIR transmission coefficient of the cover and canopy [-]
    a.tauCovCanNir = tau12(a.tauHatCovNir, a.tauHatCanNir, a.rhoCovNir, a.rhoHatCanNir)

    # NIR reflection coefficient of the cover and canopy towards the top [-]
    a.rhoCovCanNirUp = rhoUp(a.tauHatCanNir, a.rhoCovNir, a.rhoCovNir, a.rhoHatCanNir)

    # NIR reflection coefficient of the cover and canopy towards the bottom [-]
    a.rhoCovCanNirDn = rhoDn(a.tauHatCanNir, a.rhoCovNir, a.rhoHatCanNir, a.rhoHatCanNir)

    # NIR transmission coefficient of the cover, canopy and floor [-]
    a.tauCovCanFlrNir = tau12(a.tauCovCanNir, a.tauHatFlrNir, a.rhoCovCanNirDn, p.rhoFlrNir)

    # NIR reflection coefficient of the cover, canopy and floor [-]
    a.rhoCovCanFlrNir = rhoUp(a.tauCovCanNir, a.rhoCovCanNirUp, a.rhoCovCanNirDn, p.rhoFlrNir)

    # The calculated absorption coefficient equals m.a.aCanNir [-]
    # pg. 23 [1]
    a.aCanNir = 1 - a.tauCovCanFlrNir - a.rhoCovCanFlrNir

    # The calculated transmission coefficient equals m.a.aFlrNir [-]
    # pg. 23 [1]
    # addAux(gl, 'aFlrNir', gl.a.tauCovCanFlrNir)
    a.aFlrNir = a.tauCovCanFlrNir

    # NIR from the sun absorbed by the canopy [W m^{-2}]
    # Equation 32 [1]
    # addAux(gl, 'rNirSunCan', (1-p.etaGlobAir).*gl.a.aCanNir.*p.etaGlobNir.*d.iGlob)
    a.rNirSunCan = (1-p.etaGlobAir)*a.aCanNir*p.etaGlobNir*d[0]
    
    # NIR from the lamps absorbed by the canopy [W m^{-2}]
    # Equation A20 [5]
    # addAux(gl, 'rNirLampCan', p.etaLampNir.*gl.a.qLampIn.*(1-p.rhoCanNir).*(1-exp(-p.kNir*gl.a.lai)))
    a.rNirLampCan = p.etaLampNir * a.qLampIn * (1-p.rhoCanNir) * (1 - exp(-p.kNir*a.lai))

    # NIR from the interlights absorbed by the canopy [W m^{-2}]
    # Equation 7.20 [7]
    # addAux(gl, 'rNirIntLampCan', p.etaIntLampNir.*gl.a.qIntLampIn.*gl.a.fIntLampCanNir.*(1-p.rhoCanNir))
    a.rNirIntLampCan = p.etaIntLampNir * a.qIntLampIn * a.fIntLampCanNir * (1-p.rhoCanNir)


    # NIR from the sun absorbed by the floor [W m^{-2}]
    # Equation 33 [1]
    # addAux(gl, 'rNirSunFlr', (1-p.etaGlobAir).*gl.a.aFlrNir.*p.etaGlobNir.*d.iGlob)
    a.rNirSunFlr = (1-p.etaGlobAir) * a.aFlrNir * p.etaGlobNir * d[0]

    # NIR from the lamps absorbed by the floor [W m^{-2}]
    # Equation A22 [5]
    # addAux(gl, 'rNirLampFlr', (1-p.rhoFlrNir).*exp(-p.kNir*gl.a.lai).*p.etaLampNir.*gl.a.qLampIn)
    a.rNirLampFlr = (1-p.rhoFlrNir) * exp(-p.kNir*a.lai) * p.etaLampNir * a.qLampIn

    # NIR from the interlights absorbed by the floor [W m^{-2}]
    # Equation 7.21 [7]
    a.rNirIntLampFlr = p.fIntLampDown * (1-p.rhoFlrNir) * \
        exp(-p.kIntNir*a.lai*p.vIntLampPos) * \
        p.etaIntLampNir * a.qIntLampIn
    # if p.vIntLampPos==1, the lamp is above the canopy, light loses
    # exp(-k*LAI) on its way to the floor.
    # if p.vIntLampPos==0, the lamp is below the canopy, no light is
    # lost on the way to the floor
    
    # PAR from the sun absorbed by the floor [W m^{-2}]
    # Equation 34 [1]
    # addAux(gl, 'rParSunFlr', (1-p.rhoFlrPar).*exp(-p.k1Par*gl.a.lai).*gl.a.rParGhSun)
    a.rParSunFlr = (1-p.rhoFlrPar) * exp(-p.k1Par * a.lai) * a.rParGhSun

    # PAR from the lamps absorbed by the floor [W m^{-2}]
    # Equation A21 [5]
    # addAux(gl, 'rParLampFlr', (1-p.rhoFlrPar).*exp(-p.k1Par*gl.a.lai).*gl.a.rParGhLamp)
    a.rParLampFlr = (1-p.rhoFlrPar) * exp(-p.k1Par * a.lai) * a.rParGhLamp

    # PAR from the interlights absorbed by the floor [W m^{-2}]
    # Equation 7.17 [7]
    # addAux(gl, 'rParIntLampFlr', gl.a.rParGhIntLamp.*p.fIntLampDown.*(1-p.rhoFlrPar).*\
    #     exp(-p.k1IntPar*gl.a.lai.*p.vIntLampPos))
    a.rParIntLampFlr = a.rParGhIntLamp * p.fIntLampDown * (1-p.rhoFlrPar) * \
        exp(-p.k1IntPar * a.lai * p.vIntLampPos)
    
	# PAR and NIR from the lamps absorbed by the greenhouse air [W m^{-2}]
    # Equation A23 [5]
	# addAux(gl, 'rLampAir', (p.etaLampPar+p.etaLampNir)*gl.a.qLampIn - gl.a.rParLampCan - \
	# 	gl.a.rNirLampCan - gl.a.rParLampFlr - gl.a.rNirLampFlr)
    a.rLampAir = (p.etaLampPar + p.etaLampNir) * a.qLampIn - a.rParLampCan - \
        a.rNirLampCan - a.rParLampFlr - a.rNirLampFlr
	
    # PAR and NIR from the interlights absorbed by the greenhouse air [W m^{-2}]
    # Equation 7.22 [7]
	# addAux(gl, 'rIntLampAir', (p.etaIntLampPar+p.etaIntLampNir)*gl.a.qIntLampIn - gl.a.rParIntLampCan - \
	# 	gl.a.rNirIntLampCan - gl.a.rParIntLampFlr - gl.a.rNirIntLampFlr)
    a.rIntLampAir = (p.etaIntLampPar+p.etaIntLampNir)*a.qIntLampIn - a.rParIntLampCan - \
        a.rNirIntLampCan - a.rParIntLampFlr - a.rNirIntLampFlr
    
    # Global radiation from the sun absorbed by the greenhouse air [W m^{-2}]
    # Equation 35 [1]
    # addAux(gl, 'rGlobSunAir', p.etaGlobAir*d.iGlob.*\
    #     (gl.a.tauCovPar*p.etaGlobPar+(gl.a.aCanNir+gl.a.aFlrNir)*p.etaGlobNir))
    a.rGlobSunAir = p.etaGlobAir * d[0] * (a.tauCovPar * p.etaGlobPar + (a.aCanNir + a.aFlrNir) * p.etaGlobNir)
    
    # Global radiation from the sun absorbed by the cover [W m^{-2}]
    # Equation 36 [1]
    # addAux(gl, 'rGlobSunCovE', (gl.a.aCovPar*p.etaGlobPar+gl.a.aCovNir*p.etaGlobNir).*d.iGlob)
    a.rGlobSunCovE = (a.aCovPar * p.etaGlobPar + a.aCovNir * p.etaGlobNir) * d[0]

    ############################################
    #### FIR heat fluxes - Section 5.2 [1] #####
    ############################################

    # FIR transmission coefficient of the thermal screen
    # Equation 38 [1]
    # addAux(gl, 'tauThScrFirU', (1-u.thScr*(1-p.tauThScrFir)))
    a.tauThScrFirU = 1 - u[2]*(1-p.tauThScrFir)

    # FIR transmission coefficient of the blackout screen
    # addAux(gl, 'tauBlScrFirU', (1-u.blScr*(1-p.tauBlScrFir)))   
    a.tauBlScrFirU = 1 - u[7]*(1-p.tauBlScrFir)

    # Surface of canopy per floor area [-]
    # Table 3 [1]
    # addAux(gl, 'aCan', 1-exp(-p.kFir*gl.a.lai))
    a.aCan = 1 - exp(-p.kFir*a.lai)

    # FIR between greenhouse objects [W m^{-2}]
    # Table 7.4 [7]. Based on Table 3 [1] and Table A1 [5]

    # FIR between canopy and cover [W m^{-2}]
    # addAux(gl, 'rCanCovIn', fir(gl.a.aCan, p.epsCan, gl.a.epsCovFir, \
    #     p.tauLampFir*gl.a.tauThScrFirU.*gl.a.tauBlScrFirU,\
    #     x.tCan, x.tCovIn))
    a.rCanCovIn = fir(a.aCan, p.epsCan, a.epsCovFir, p.tauLampFir*a.tauThScrFirU*a.tauBlScrFirU, x[4], x[5], p.sigma)

    # FIR between canopy and sky [W m^{-2}]
    # addAux(gl, 'rCanSky', fir(gl.a.aCan, p.epsCan, p.epsSky, \
    #     p.tauLampFir*gl.a.tauCovFir.*gl.a.tauThScrFirU.*gl.a.tauBlScrFirU,\
    #     x.tCan, d.tSky))
    a.rCanSky = fir(a.aCan, p.epsCan, p.epsSky, p.tauLampFir*a.tauCovFir*a.tauThScrFirU*a.tauBlScrFirU, x[4], d[5], p.sigma)

    # FIR between canopy and thermal screen [W m^{-2}]
    # addAux(gl, 'rCanThScr', fir(gl.a.aCan, p.epsCan, p.epsThScrFir, \
    #     p.tauLampFir*u.thScr.*gl.a.tauBlScrFirU, x.tCan, x.tThScr))
    a.rCanThScr = fir(a.aCan, p.epsCan, p.epsThScrFir, \
        p.tauLampFir * u[2] * a.tauBlScrFirU, x[4], x[7], p.sigma)

    # FIR between canopy and floor [W m^{-2}]
    # addAux(gl, 'rCanFlr', fir(gl.a.aCan, p.epsCan, p.epsFlr, \
    #     p.fCanFlr, x.tCan, x.tFlr))
    a.rCanFlr = fir(a.aCan, p.epsCan, p.epsFlr, p.fCanFlr, x[4], x[8], p.sigma)
    # print("aCan", a.aCan)
    # print("epsCan", p.epsCan)
    # print("epsFlr", p.epsFlr)
    # print("fCanFlr", p.fCanFlr)
    # print("tCan", x[4])
    # print("tFlr", x[8])
    # print("sigma", p.sigma)
    # FIR between pipes and cover [W m^{-2}]
    # addAux(gl, 'rPipeCovIn', fir(p.aPipe, p.epsPipe, gl.a.epsCovFir, \
    #     p.tauIntLampFir*p.tauLampFir*gl.a.tauThScrFirU.*gl.a.tauBlScrFirU*0.49.*\
    #     exp(-p.kFir*gl.a.lai), x.tPipe, x.tCovIn))
    a.rPipeCovIn = fir(p.aPipe, p.epsPipe, a.epsCovFir, \
        p.tauIntLampFir*p.tauLampFir*a.tauThScrFirU*a.tauBlScrFirU*0.49* \
        exp(-p.kFir*a.lai), x[9], x[5], p.sigma)

    # FIR between pipes and sky [W m^{-2}]
    # addAux(gl, 'rPipeSky', fir(p.aPipe, p.epsPipe, p.epsSky, \
    #     p.tauIntLampFir*p.tauLampFir*gl.a.tauCovFir.*gl.a.tauThScrFirU.*\
    #     gl.a.tauBlScrFirU*0.49.*exp(-p.kFir*gl.a.lai), x.tPipe, d.tSky))
    a.rPipeSky = fir(p.aPipe, p.epsPipe, p.epsSky, \
        p.tauIntLampFir*p.tauLampFir*a.tauCovFir*a.tauThScrFirU*0.49* \
        exp(-p.kFir*a.lai), x[9], d[5], p.sigma)

    # FIR between pipes and thermal screen [W m^{-2}]
    # addAux(gl, 'rPipeThScr', fir(p.aPipe, p.epsPipe, p.epsThScrFir, \
    #     p.tauIntLampFir*p.tauLampFir*u.thScr.*gl.a.tauBlScrFirU*0.49.*\
    #     exp(-p.kFir*gl.a.lai), x.tPipe, x.tThScr))
    a.rPipeThScr = fir(p.aPipe, p.epsPipe, p.epsThScrFir,
        p.tauIntLampFir*p.tauLampFir*u[2]*a.tauBlScrFirU*0.49* \
        exp(-p.kFir*a.lai), x[9], x[7], p.sigma)

    # FIR between pipes and floor [W m^{-2}]
    # addAux(gl, 'rPipeFlr', fir(p.aPipe, p.epsPipe, p.epsFlr, 0.49, x.tPipe, x.tFlr))
    a.rPipeFlr = fir(p.aPipe, p.epsPipe, p.epsFlr, 0.49, x[9], x[8], p.sigma)

    # FIR between pipes and canopy [W m^{-2}]
    # addAux(gl, 'rPipeCan', fir(p.aPipe, p.epsPipe, p.epsCan, \
    #     0.49.*(1-exp(-p.kFir*gl.a.lai)), x.tPipe, x.tCan))
    a.rPipeCan = fir(p.aPipe, p.epsPipe, p.epsCan, \
        0.49*(1-exp(-p.kFir*a.lai)), x[9], x[4], p.sigma)

    # FIR between floor and cover [W m^{-2}]
    # fir(1, p.epsFlr, gl.a.epsCovFir, \
    #     p.tauIntLampFir*p.tauLampFir*gl.a.tauThScrFirU.*gl.a.tauBlScrFirU*\
    #     (1-0.49*pi*p.lPipe*p.phiPipeE).*exp(-p.kFir*gl.a.lai), x.tFlr, x.tCovIn)
    a.rFlrCovIn = fir(1, p.epsFlr, a.epsCovFir, \
        p.tauIntLampFir*p.tauLampFir*a.tauThScrFirU*a.tauBlScrFirU* \
        (1 - 0.49*M_PI*p.lPipe*p.phiPipeE) * exp(-p.kFir*a.lai), x[8], x[5], p.sigma)
    # print("epsFlr", p.epsFlr)
    # print("epsCovFir", a.epsCovFir)
    # print("tauIntLampFir", p.tauIntLampFir)
    # print("tauLampFir", p.tauLampFir)
    # print("tauThScrFirU", a.tauThScrFirU)
    # print("tauBlScrFirU", a.tauBlScrFirU)
    # print("lPipe", p.lPipe)
    # print("phiPipeE", p.phiPipeE)
    # print("kFir", p.kFir)
    # print("lai", a.lai)
    # print("tCovIn", x[5])

    # FIR between floor and sky [W m^{-2}]
    # addAux(gl, 'rFlrSky', fir(1, p.epsFlr, p.epsSky, \
    #     p.tauIntLampFir*p.tauLampFir*gl.a.tauCovFir.*gl.a.tauThScrFirU.*gl.a.tauBlScrFirU*\
    #     (1-0.49*pi*p.lPipe*p.phiPipeE).*exp(-p.kFir*gl.a.lai), x.tFlr, d.tSky))
    a.rFlrSky = fir(1, p.epsFlr, p.epsSky, \
        p.tauIntLampFir*p.tauLampFir*a.tauCovFir*a.tauThScrFirU*a.tauBlScrFirU* \
        (1 - 0.49*M_PI*p.lPipe*p.phiPipeE) * exp(-p.kFir*a.lai), x[8], d[5], p.sigma)

    # FIR between floor and thermal screen [W m^{-2}]
    # addAux(gl, 'rFlrThScr', fir(1, p.epsFlr, p.epsThScrFir, \
    #     p.tauIntLampFir*p.tauLampFir*u.thScr.*gl.a.tauBlScrFirU*(1-0.49*pi*p.lPipe*p.phiPipeE).*\
    #     exp(-p.kFir*gl.a.lai), x.tFlr, x.tThScr))
    a.rFlrThScr = fir(1, p.epsFlr, p.epsThScrFir, \
        p.tauIntLampFir*p.tauLampFir*u[2]*a.tauBlScrFirU * (1 - 0.49*M_PI*p.lPipe*p.phiPipeE)* \
        exp(-p.kFir*a.lai), x[8], x[7], p.sigma)

    # FIR between thermal screen and cover [W m^{-2}]
    # addAux(gl, 'rThScrCovIn', fir(1, p.epsThScrFir, gl.a.epsCovFir, \
    #     u.thScr, x.tThScr, x.tCovIn))
    a.rThScrCovIn = fir(1, p.epsThScrFir, a.epsCovFir, \
        u[2], x[7], x[5], p.sigma)

    # FIR between thermal screen and sky [W m^{-2}]
    # addAux(gl, 'rThScrSky', fir(1, p.epsThScrFir, p.epsSky, \
    #     gl.a.tauCovFir.*u.thScr, x.tThScr, d.tSky))
    a.rThScrSky = fir(1, p.epsThScrFir, p.epsSky, \
        a.tauCovFir*u[2], x[7], d[5], p.sigma)

    # FIR between cover and sky [W m^{-2}]
    # addAux(gl, 'rCovESky', fir(1, gl.a.aCovFir, p.epsSky, 1, x.tCovE, d.tSky))
    a.rCovESky = fir(1, a.aCovFir, p.epsSky, 1, x[6], d[5], p.sigma)

    # FIR between lamps and floor [W m^{-2}]
    # addAux(gl, 'rFirLampFlr', fir(p.aLamp, p.epsLampBottom, p.epsFlr, \
    #     p.tauIntLampFir.*(1-0.49*pi*p.lPipe*p.phiPipeE).*exp(-p.kFir*gl.a.lai), x.tLamp, x.tFlr))
    a.rFirLampFlr = fir(p.aLamp, p.epsLampBottom, p.epsFlr, \
        p.tauIntLampFir * (1 - 0.49*M_PI*p.lPipe*p.phiPipeE) * exp(-p.kFir*a.lai), x[17], x[8], p.sigma)

    # FIR between lamps and pipe [W m^{-2}]
    # addAux(gl, 'rLampPipe', fir(p.aLamp, p.epsLampBottom, p.epsPipe, \
    #     p.tauIntLampFir.*0.49*pi*p.lPipe*p.phiPipeE.*exp(-p.kFir*gl.a.lai), x.tLamp, x.tPipe))
    a.rLampPipe = fir(p.aLamp, p.epsLampBottom, p.epsPipe, \
        p.tauIntLampFir*0.49*M_PI*p.lPipe*p.phiPipeE*exp(-p.kFir*a.lai), x[17], x[9], p.sigma)

    # FIR between lamps and canopy [W m^{-2}]
    # addAux(gl, 'rFirLampCan', fir(p.aLamp, p.epsLampBottom, p.epsCan, \
    #     gl.a.aCan, x.tLamp, x.tCan))
    a.rFirLampCan = fir(p.aLamp, p.epsLampBottom, p.epsCan, \
        a.aCan, x[17], x[4], p.sigma)

    # FIR between lamps and thermal screen [W m^{-2}]
    # addAux(gl, 'rLampThScr', fir(p.aLamp, p.epsLampTop, p.epsThScrFir, \
    #     u.thScr.*gl.a.tauBlScrFirU, x.tLamp, x.tThScr))
    a.rLampThScr = fir(p.aLamp, p.epsLampTop, p.epsThScrFir, \
        u[2]*a.tauBlScrFirU, x[17], x[7], p.sigma)

    # FIR between lamps and cover [W m^{-2}]
    # addAux(gl, 'rLampCovIn', fir(p.aLamp, p.epsLampTop, gl.a.epsCovFir, \
    #     gl.a.tauThScrFirU.*gl.a.tauBlScrFirU, x.tLamp, x.tCovIn))
    a.rLampCovIn = fir(p.aLamp, p.epsLampTop, a.epsCovFir, \
        a.tauThScrFirU*a.tauBlScrFirU, x[17], x[5], p.sigma)

    # FIR between lamps and sky [W m^{-2}]
    # addAux(gl, 'rLampSky', fir(p.aLamp, p.epsLampTop, p.epsSky, \
    #     gl.a.tauCovFir.*gl.a.tauThScrFirU.*gl.a.tauBlScrFirU, x.tLamp, d.tSky))
    a.rLampSky = fir(p.aLamp, p.epsLampTop, p.epsSky, \
        a.tauCovFir*a.tauThScrFirU*a.tauBlScrFirU, x[17], d[5], p.sigma)

    # FIR between grow pipes and canopy [W m^{-2}]
    # addAux(gl, 'rGroPipeCan', fir(p.aGroPipe, p.epsGroPipe, p.epsCan, 1, x.tGroPipe, x.tCan))
    a.rGroPipeCan = fir(p.aGroPipe, p.epsGroPipe, p.epsCan, 1, x[19], x[4], p.sigma)

    # FIR between blackout screen and floor [W m^{-2}]	
    # addAux(gl, 'rFlrBlScr', fir(1, p.epsFlr, p.epsBlScrFir, \
    #     p.tauIntLampFir*p.tauLampFir*u.blScr*(1-0.49*pi*p.lPipe*p.phiPipeE).*\
    #     exp(-p.kFir*gl.a.lai), x.tFlr, x.tBlScr))
    a.rFlrBlScr = fir(1, p.epsFlr, p.epsBlScrFir, \
        p.tauIntLampFir*p.tauLampFir*u[7] * (1 - 0.49*M_PI*p.lPipe*p.phiPipeE)* \
        exp(-p.kFir*a.lai), x[8], x[20], p.sigma)

    # FIR between blackout screen and pipe [W m^{-2}]
    # addAux(gl, 'rPipeBlScr', fir(p.aPipe, p.epsPipe, p.epsBlScrFir, \
    #     p.tauIntLampFir*p.tauLampFir*u.blScr*0.49.*exp(-p.kFir*gl.a.lai), x.tPipe, x.tBlScr))
    a.rPipeBlScr = fir(p.aPipe, p.epsPipe, p.epsBlScrFir, \
        p.tauIntLampFir*p.tauLampFir*u[7]*0.49*exp(-p.kFir*a.lai), x[9], x[20], p.sigma)

    # FIR between blackout screen and canopy [W m^{-2}]
    # addAux(gl, 'rCanBlScr', fir(gl.a.aCan, p.epsCan, p.epsBlScrFir, \
    #     p.tauLampFir*u.blScr, x.tCan, x.tBlScr))
    a.rCanBlScr = fir(a.aCan, p.epsCan, p.epsBlScrFir, \
        p.tauLampFir*u[7], x[4], x[20], p.sigma)

    # FIR between blackout screen and thermal screen [W m^{-2}]
    # addAux(gl, 'rBlScrThScr', fir(u.blScr, p.epsBlScrFir, \
    #     p.epsThScrFir, u.thScr, x.tBlScr, x.tThScr))
    a.rBlScrThScr = fir(u[7], p.epsBlScrFir, \
        p.epsThScrFir, u[2], x[20], x[7], p.sigma)

    # FIR between blackout screen and cover [W m^{-2}]
    # addAux(gl, 'rBlScrCovIn', fir(u.blScr, p.epsBlScrFir, gl.a.epsCovFir, \
    #     gl.a.tauThScrFirU, x.tBlScr, x.tCovIn))
    a.rBlScrCovIn = fir(u[7], p.epsBlScrFir, a.epsCovFir, \
        a.tauThScrFirU, x[20], x[5], p.sigma)

    # FIR between blackout screen and sky [W m^{-2}]
    # addAux(gl, 'rBlScrSky', fir(u.blScr, p.epsBlScrFir, p.epsSky, \
    #     gl.a.tauCovFir.*gl.a.tauThScrFirU, x.tBlScr, d.tSky))
    a.rBlScrSky = fir(u[7], p.epsBlScrFir, p.epsSky, \
        a.tauCovFir*a.tauThScrFirU, x[20], d[5], p.sigma)

    # FIR between blackout screen and lamps [W m^{-2}]
    # addAux(gl, 'rLampBlScr', fir(p.aLamp, p.epsLampTop, p.epsBlScrFir, \
    #     u.blScr, x.tLamp, x.tBlScr))
    a.rLampBlScr = fir(p.aLamp, p.epsLampTop, p.epsBlScrFir, \
        u[7], x[17], x[20], p.sigma)

    # Fraction of radiation going up from the interlight to the canopy [-]
    # Equation 7.29 [7]
    # addAux(gl, 'fIntLampCanUp', 1-exp(-p.kIntFir*(1-p.vIntLampPos).*gl.a.lai))
    a.fIntLampCanUp = 1 - exp(-p.kIntFir * (1-p.vIntLampPos) * a.lai)

    # Fraction of radiation going down from the interlight to the canopy [-]
    # Equation 7.30 [7]
    # addAux(gl, 'fIntLampCanDown', 1-exp(-p.kIntFir*p.vIntLampPos.*gl.a.lai))
    a.fIntLampCanDown = 1 - exp(-p.kIntFir * p.vIntLampPos * a.lai)

    # FIR between interlights and floor [W m^{-2}]
    # addAux(gl, 'rFirIntLampFlr', fir(p.aIntLamp, p.epsIntLamp, p.epsFlr, \
    #     (1-0.49*pi*p.lPipe*p.phiPipeE).*(1-gl.a.fIntLampCanDown),\
    #     x.tIntLamp, x.tFlr))
    a.rFirIntLampFlr = fir(p.aIntLamp, p.epsIntLamp, p.epsFlr, \
        (1 - 0.49*M_PI*p.lPipe*p.phiPipeE) * (1-a.fIntLampCanDown),\
        x[18], x[8], p.sigma)

    # FIR between interlights and pipe [W m^{-2}]
    # addAux(gl, 'rIntLampPipe', fir(p.aIntLamp, p.epsIntLamp, p.epsPipe, \
    #     0.49*pi*p.lPipe*p.phiPipeE.*(1-gl.a.fIntLampCanDown),\
    #     x.tIntLamp, x.tPipe))
    a.rIntLampPipe = fir(p.aIntLamp, p.epsIntLamp, p.epsPipe, \
        0.49*M_PI*p.lPipe*p.phiPipeE * (1-a.fIntLampCanDown),\
        x[18], x[9], p.sigma)

    # FIR between interlights and canopy [W m^{-2}]
    # addAux(gl, 'rFirIntLampCan', fir(p.aIntLamp, p.epsIntLamp, p.epsCan, \
    #     gl.a.fIntLampCanDown+gl.a.fIntLampCanUp, x.tIntLamp, x.tCan))
    a.rFirIntLampCan = fir(p.aIntLamp, p.epsIntLamp, p.epsCan, \
        a.fIntLampCanDown+a.fIntLampCanUp, x[18], x[4], p.sigma)

    # FIR between interlights and toplights [W m^{-2}]
    # addAux(gl, 'rIntLampLamp', fir(p.aIntLamp, p.epsIntLamp, p.epsLampBottom, \
    #     (1-gl.a.fIntLampCanUp).*p.aLamp, x.tIntLamp, x.tLamp))
    a.rIntLampLamp = fir(p.aIntLamp, p.epsIntLamp, p.epsLampBottom, \
        (1-a.fIntLampCanUp) * p.aLamp, x[18], x[17], p.sigma)

    # FIR between interlights and blackout screen [W m^{-2}]
    # addAux(gl, 'rIntLampBlScr', fir(p.aIntLamp, p.epsIntLamp, p.epsBlScrFir, \
    #     u.blScr.*p.tauLampFir.*(1-gl.a.fIntLampCanUp), x.tIntLamp, x.tBlScr))
    a.rIntLampBlScr = fir(p.aIntLamp, p.epsIntLamp, p.epsBlScrFir, \
        u[7]*p.tauLampFir * (1-a.fIntLampCanUp), x[18], x[20], p.sigma)
        # if p.vIntLampPos==0, the lamp is above the canopy, no light is
        # lost on its way up
        # if p.vIntLampPos==1, the lamp is below the canopy, the light
        # loses exp(-k*LAI) on its way up

    # FIR between interlights and thermal screen [W m^{-2}]
    # addAux(gl, 'rIntLampThScr', fir(p.aIntLamp, p.epsIntLamp, p.epsThScrFir, \
    #     u.thScr.*gl.a.tauBlScrFirU.*p.tauLampFir.*(1-gl.a.fIntLampCanUp),\
    #     x.tIntLamp, x.tThScr))
    a.rIntLampThScr = fir(p.aIntLamp, p.epsIntLamp, p.epsThScrFir, \
        u[2]*a.tauBlScrFirU*p.tauLampFir * (1-a.fIntLampCanUp),\
        x[18], x[7], p.sigma)

    # FIR between interlights and cover [W m^{-2}]
    # addAux(gl, 'rIntLampCovIn', fir(p.aIntLamp, p.epsIntLamp, gl.a.epsCovFir, \
    #     gl.a.tauThScrFirU.*gl.a.tauBlScrFirU.*p.tauLampFir.*(1-gl.a.fIntLampCanUp),\
    #     x.tIntLamp, x.tCovIn))
    a.rIntLampCovIn = fir(p.aIntLamp, p.epsIntLamp, a.epsCovFir, \
        a.tauThScrFirU*a.tauBlScrFirU*p.tauLampFir * (1-a.fIntLampCanUp),\
        x[18], x[5], p.sigma)

    # FIR between interlights and sky [W m^{-2}]
    # addAux(gl, 'rIntLampSky', fir(p.aIntLamp, p.epsIntLamp, p.epsSky, \
    #     gl.a.tauCovFir.*gl.a.tauThScrFirU.*gl.a.tauBlScrFirU.*p.tauLampFir.*(1-gl.a.fIntLampCanUp),\
    #     x.tIntLamp, d.tSky))
    a.rIntLampSky = fir(p.aIntLamp, p.epsIntLamp, p.epsSky, \
        a.tauCovFir*a.tauThScrFirU*a.tauBlScrFirU*p.tauLampFir * (1-a.fIntLampCanUp),\
        x[18], d[5], p.sigma)

    #############################
    #### Natural Ventilation ####
    #############################

    # Aperature of the roof
        # Aperture of the roof [m^{2}]
    # Equation 67 [1]
    # addAux(gl, 'aRoofU', u.roof*p.aRoof)
    # addAux(gl, 'aRoofUMax', p.aRoof)
    # addAux(gl, 'aRoofMin', DynamicElement('0',0))
    a.aRoofU = u[3]*p.aRoof
    a.aRoofUMax = p.aRoof
    a.aRoofMin = 0

    # Aperture of the sidewall [m^{2}]
    # Equation 68 [1] 
    # (this is 0 in the Dutch greenhouse)
    # addAux(gl, 'aSideU', u.side*p.aSide)
    a.aSideU = u[10]*p.aSide

    # Ratio between roof vent area and total ventilation area [-]
    # (not very clear in the reference [1], but always 1 if m.a.aSideU == 0)
    #     addAux(m, 'etaRoof', m.a.aRoofU./max(m.a.aRoofU+m.a.aSideU,0.01)) 
    # addAux(gl, 'etaRoof', '1') 
    # addAux(gl, 'etaRoofNoSide', '1')
    a.etaRoof = 1
    a.etaRoofNoSide = 1

    # Ratio between side vent area and total ventilation area [-]
    # (not very clear in the reference [1], but always 0 if m.a.aSideU == 0)    
    # addAux(gl, 'etaSide', '0')
    a.etaSide = 0

    # Discharge coefficient [-]
    # Equation 73 [1]
    # addAux(gl, 'cD', p.cDgh*(1-p.etaShScrCd*u.shScr))
    a.cD = p.cDgh * (1 - p.etaShScrCd*u[8])

    # Discharge coefficient [-]
    # Equation 74 [-]
    # addAux(gl, 'cW', p.cWgh*(1-p.etaShScrCw*u.shScr))
    a.cW = p.cWgh * (1 - p.etaShScrCw*u[8])

    # Natural ventilation rate due to roof ventilation [m^{3} m^{-2} s^{-1}]
    # Equation 64 [1]
    # addAux(gl, 'fVentRoof2', u.roof*p.aRoof.*gl.a.cD/(2.*p.aFlr).*\
    #     sqrt(abs(p.g*p.hVent*(x.tAir-d.tOut)./(2*(0.5*x.tAir+0.5*d.tOut+273.15))+gl.a.cW.*d.wind.^2)))
    a.fVentRoof2 = u[3] * p.aRoof * a.cD/(2*p.aFlr) * \
        sqrt(fabs(p.g*p.hVent * (x[2] - d[1]) / (2*(0.5*x[2] + 0.5*d[1] + 273.15)) + a.cW * d[4]**2))

    # addAux(gl, 'fVentRoof2Max', p.aRoof.*gl.a.cD/(2.*p.aFlr).*\
    #     sqrt(abs(p.g*p.hVent*(x.tAir-d.tOut)./(2*(0.5*x.tAir+0.5*d.tOut+273.15)) + gl.a.cW.*d.wind.^2)))
    a.fVentRoof2Max = p.aRoof * a.cD/(2*p.aFlr) * \
        sqrt(fabs(p.g*p.hVent * (x[2]-d[1]) / (2*(0.5*x[2] + 0.5*d[1] + 273.15)) + a.cW*d[4]**2))
    
    # addAux(gl, 'fVentRoof2Min', DynamicElement('0',0))
    a.fVentRoof2Min = 0

    # Ventilation rate through roof and side vents [m^{3} m^{-2} s^{-1}]
    # Equation 65 [1]
    # addAux(gl, 'fVentRoofSide2', gl.a.cD/p.aFlr.*sqrt(\
    #     (gl.a.aRoofU.*gl.a.aSideU./sqrt(fmax(gl.a.aRoofU.^2+gl.a.aSideU.^2,0.01))).^2 .* \
    #     (2*p.g*p.hSideRoof*(x.tAir-d.tOut)./(0.5*x.tAir+0.5*d.tOut+273.15))+\
    #     ((gl.a.aRoofU+gl.a.aSideU)/2).^2.*gl.a.cW.*d.wind.^2))

    a.fVentRoofSide2 = a.cD / p.aFlr * sqrt(\
        (a.aRoofU*a.aSideU / sqrt(fmax(a.aRoofU**2 + a.aSideU**2, 0.01)))**2 *\
        (2*p.g*p.hSideRoof*(x[2]-d[1])/(0.5*x[2] + +0.5*d[1] +273.15)) + \
        ((a.aRoofU + a.aSideU/2)**2 * a.cW * d[4]**2))

    # Ventilation rate through sidewall only [m^{3} m^{-2} s^{-1}]
    # Equation 66 [1]
    # addAux(gl, 'fVentSide2', gl.a.cD.*gl.a.aSideU.*d.wind/(2*p.aFlr).*sqrt(gl.a.cW))
    a.fVentSide2 = a.cD * a.aSideU * d[4] / (2*p.aFlr) * sqrt(a.cW)

    # Leakage ventilation [m^{3} m^{-2} s^{-1}]
    # Equation 70 [1]
    # addAux(gl, 'fLeakage', ifElse('d.wind<p.minWind',p.minWind*p.cLeakage,p.cLeakage*d.wind))
    if d[4] < p.minWind:
        a.fLeakage = p.minWind * p.cLeakage
    else:
        a.fLeakage = p.cLeakage * d[4]

    # # Total ventilation through the roof [m^{3} m^{-2} s^{-1}]
    # # Equation 71 [1], Equation A42 [5]
    # addAux(gl, 'fVentRoof', ifElse([getDefStr(gl.a.etaRoof) '>=p.etaRoofThr'], p.etaInsScr*gl.a.fVentRoof2+p.cLeakTop*gl.a.fLeakage,\
    #     p.etaInsScr*(max(u.thScr,u.blScr).*gl.a.fVentRoof2+(1-max(u.thScr,u.blScr)).*gl.a.fVentRoofSide2.*gl.a.etaRoof)\
    #     +p.cLeakTop*gl.a.fLeakage))
    if a.etaRoof >= p.etaRoofThr:
        a.fVentRoof = p.etaInsScr * a.fVentRoof2 + p.cLeakTop * a.fLeakage
    else:
        a.fVentRoof = p.etaInsScr * (fmax(u[2], u[7]) * a.fVentRoof2 + (1 - fmax(u[2], u[7])) * a.fVentRoofSide2 * a.etaRoof) \
            + p.cLeakTop * a.fLeakage

    # # Total ventilation through side vents [m^{3} m^{-2} s^{-1}]
    # # Equation 72 [1], Equation A43 [5]
    # addAux(gl, 'fVentSide', ifElse([getDefStr(gl.a.etaRoof) '>=p.etaRoofThr'],p.etaInsScr*gl.a.fVentSide2+(1-p.cLeakTop)*gl.a.fLeakage,\
    #     p.etaInsScr*(max(u.thScr,u.blScr).*gl.a.fVentSide2+(1-max(u.thScr,u.blScr)).*gl.a.fVentRoofSide2.*gl.a.etaSide)\
    #     +(1-p.cLeakTop)*gl.a.fLeakage))

    if a.etaRoof >= p.etaRoofThr:
        a.fVentSide = p.etaInsScr * a.fVentSide2 + (1 - p.cLeakTop) * a.fLeakage
    else:
        a.fVentSide = p.etaInsScr * (fmax(u[2], u[7]) * a.fVentSide2 + (1 - fmax(u[2], u[7])) * a.fVentRoofSide2 * a.etaSide) \
            + (1 - p.cLeakTop) * a.fLeakage

    #######################
    #### Control Rules ####
    #######################
    # CO2 concentration in main compartment [ppm]
    a.co2InPpm = co2dens2ppm(x[2], 1e-6*x[0])

    # Relative humidity [%]
    a.rhIn = 100*x[15]/satVp(x[2])


    ###################################
    #### Convection and conduction ####
    ###################################

    # density of air as it depends on pressure and temperature, see
    # https://en.wikipedia.org/wiki/Density_of_air
    a.rhoTop = p.mAir * p.pressure / ((x[3] + 273.15) * p.R)
    a.rhoAir = p.mAir * p.pressure / ((x[2] + 273.15) * p.R)

    # See [4], where rhoMean is "the mean
    # density of air beneath and above the screen".
    a.rhoAirMean = 0.5 * (a.rhoTop + a.rhoAir)

    # Air flux through the thermal screen [m s^{-1}]
    # Equation 40 [1], Equation A36 [5]
    # There is a mistake in [1], see equation 5.68, pg. 91, [4]
    # tOut, rhoOut, should be tTop, rhoTop
    # There is also a mistake in [4], whenever sqrt is taken, abs should be included
    # addAux(gl, 'fThScr', u.thScr*p.kThScr.*(abs((x.tAir-x.tTop)).^0.66) + \ 
    #     ((1-u.thScr)./gl.a.rhoAirMean).*sqrt(0.5*gl.a.rhoAirMean.*(1-u.thScr).*p.g.*abs(gl.a.rhoAir-gl.a.rhoTop)))
    a.fThScr = u[2] * p.kThScr * (fabs(x[2] - x[3])**0.66) + \
        ((1 - u[2]) / a.rhoAirMean) * sqrt(0.5 * a.rhoAirMean * (1 - u[2]) * p.g * fabs(a.rhoAir - a.rhoTop))
    # Air flux through the blackout screen [m s^{-1}]
    # Equation A37 [5]
    # addAux(gl, 'fBlScr', u.blScr*p.kBlScr.*(abs((x.tAir-x.tTop)).^0.66) + \ 
    #     ((1-u.blScr)./gl.a.rhoAirMean).*sqrt(0.5*gl.a.rhoAirMean.*(1-u.blScr).*p.g.*abs(gl.a.rhoAir-gl.a.rhoTop)))

    a.fBlScr = u[7] * p.kBlScr * (fabs(x[2] - x[3])**0.66) + \
        ((1 - u[7]) / a.rhoAirMean) * sqrt(0.5 * a.rhoAirMean * (1 - u[7]) * p.g * fabs(a.rhoAir - a.rhoTop))

    # Air flux through the screens [m s^{-1}]
    # Equation A38 [5]
    # addAux(gl, 'fScr', min(gl.a.fThScr,gl.a.fBlScr))
    a.fScr = fmin(a.fThScr, a.fBlScr)

    ##########################################################
    #### Convective and conductive heat fluxes [W m^{-2}] ####
    ##########################################################

    # # Forced ventilation (doesn't exist in current gh)
    # addAux(gl, 'fVentForced', DynamicElement('0', 0))
    a.fVentForced = 0
    # # Between canopy and air in main compartment [W m^{-2}]
    # addAux(gl, 'hCanAir', sensible(2*p.alfaLeafAir*gl.a.lai, x.tCan, x.tAir))
    a.hCanAir = sensible(2*p.alfaLeafAir*a.lai, x[4], x[2])

    # # Between air in main compartment and floor [W m^{-2}]
    # addAux(gl, 'hAirFlr', 
    # sensible(ifElse('x.tFlr>x.tAir',1.7*nthroot(abs(x.tFlr-x.tAir),3),1.3*nthroot(abs(x.tAir-x.tFlr),4)), x.tAir,x.tFlr)
    if x[8] > x[2]:
        a.hAirFlr = sensible(1.7 * fabs(x[8] - x[2])**(1/3), x[2], x[8])
    else:
        a.hAirFlr = sensible(1.3 * fabs(x[2] - x[8])**(1/4), x[2], x[8])

    # # Between air in main compartment and thermal screen [W m^{-2}]
    # addAux(gl, 'hAirThScr', sensible(1.7.*u.thScr.*nthroot(abs(x.tAir-x.tThScr),3),\
    #     x.tAir,x.tThScr))
    a.hAirThScr = sensible(1.7 * u[2] * fabs(x[2] - x[7])**(1/3), x[2], x[7])

    # # Between air in main compartment and blackout screen [W m^{-2}]
    # # Equations A28, A32 [5]
    # addAux(gl, 'hAirBlScr', sensible(1.7.*u.blScr.*nthroot(abs(x.tAir-x.tBlScr),3),\
    #     x.tAir,x.tBlScr))
    a.hAirBlScr = sensible(1.7 * u[7] * fabs(x[2] - x[20])**(1/3), x[2], x[20])
        
    # # Between air in main compartment and outside air [W m^{-2}]
    # addAux(gl, 'hAirOut', sensible(p.rhoAir*p.cPAir*(gl.a.fVentSide+gl.a.fVentForced),\
    #     x.tAir, d.tOut))
    a.hAirOut = sensible(p.rhoAir * p.cPAir * (a.fVentSide + a.fVentForced), \
        x[2], d[1])
        
    # # Between air in main and top compartment [W m^{-2}]
    # addAux(gl, 'hAirTop', sensible(p.rhoAir*p.cPAir*gl.a.fScr, x.tAir, x.tTop))
    a.hAirTop = sensible(p.rhoAir * p.cPAir * a.fScr, x[2], x[3])

    # # Between thermal screen and top compartment [W m^{-2}]
    # addAux(gl, 'hThScrTop', sensible(1.7.*u.thScr.*nthroot(abs(x.tThScr-x.tTop),3),\
    #     x.tThScr,x.tTop))
    a.hThScrTop = sensible(1.7 * u[2] * fabs(x[7] - x[3])**(1/3), x[7], x[3])

    # # Between blackout screen and top compartment [W m^{-2}]
    # addAux(gl, 'hBlScrTop', sensible(1.7.*u.blScr.*nthroot(abs(x.tBlScr-x.tTop),3),\
    #     x.tBlScr,x.tTop))
    a.hBlScrTop = sensible(1.7 * u[7] * fabs(x[20] - x[3])**(1/3), x[20], x[3])

    # # Between top compartment and cover [W m^{-2}]
    # addAux(gl, 'hTopCovIn', sensible(p.cHecIn*nthroot(abs(x.tTop-x.tCovIn),3)*p.aCov/p.aFlr,\
    #     x.tTop,x.tCovIn))
    a.hTopCovIn = sensible(p.cHecIn * p.aCov/p.aFlr * fabs(x[3] - x[5])**(1/3), \
        x[3], x[5])

    # # Between top compartment and outside air [W m^{-2}]
    # addAux(gl, 'hTopOut', sensible(p.rhoAir*p.cPAir*gl.a.fVentRoof, x.tTop, d.tOut))
    a.hTopOut = sensible(p.rhoAir * p.cPAir * a.fVentRoof, x[3], d[1])

    # # Between cover and outside air [W m^{-2}]
    # addAux(gl, 'hCovEOut', sensible(\
    #     p.aCov/p.aFlr*(p.cHecOut1+p.cHecOut2*d.wind.^p.cHecOut3),\
    #     x.tCovE, d.tOut))
    a.hCovEOut = sensible(\
        p.aCov/p.aFlr * (p.cHecOut1 + p.cHecOut2 * d[4]**p.cHecOut3),\
        x[6], d[1])

    # # Between pipes and air in main compartment [W m^{-2}]
    # addAux(gl, 'hPipeAir', sensible(\
    #     1.99*pi*p.phiPipeE*p.lPipe*(abs(x.tPipe-x.tAir)).^0.32,\
    #     x.tPipe, x.tAir))
    a.hPipeAir = sensible(\
        1.99 * M_PI * p.phiPipeE * p.lPipe * fabs(x[9] - x[2])**0.32,\
        x[9], x[2])
        
    # # Between floor and soil layer 1 [W m^{-2}]
    # addAux(gl, 'hFlrSo1', sensible(\
    #     2/(p.hFlr/p.lambdaFlr+p.hSo1/p.lambdaSo),\
    #     x.tFlr, x.tSo1))
    a.hFlrSo1 = sensible(\
        2/(p.hFlr/p.lambdaFlr + p.hSo1/p.lambdaSo),\
        x[8], x[10])

    # # Between soil layers 1 and 2 [W m^{-2}]
    # addAux(gl, 'hSo1So2', sensible(2*p.lambdaSo/(p.hSo1+p.hSo2),\
    #     x.tSo1, x.tSo2))
    a.hSo1So2 = sensible(2*p.lambdaSo/(p.hSo1+p.hSo2),\
        x[10], x[11])

    # # Between soil layers 2 and 3 [W m^{-2}]
    # addAux(gl, 'hSo2So3', sensible(2*p.lambdaSo/(p.hSo2+p.hSo3), x.tSo2, x.tSo3))
    a.hSo2So3 = sensible(2*p.lambdaSo/(p.hSo2+p.hSo3), x[11], x[12])

    # # Between soil layers 3 and 4 [W m^{-2}]
    # addAux(gl, 'hSo3So4', sensible(2*p.lambdaSo/(p.hSo3+p.hSo4), x.tSo3, x.tSo4))
    a.hSo3So4 = sensible(2*p.lambdaSo/(p.hSo3+p.hSo4), x[12], x[13])

    # # Between soil layers 4 and 5 [W m^{-2}]
    # addAux(gl, 'hSo4So5', sensible(2*p.lambdaSo/(p.hSo4+p.hSo5), x.tSo4, x.tSo5))
    a.hSo4So5 = sensible(2*p.lambdaSo/(p.hSo4+p.hSo5), x[13], x[14])

    # # Between soil layer 5 and the external soil temperature [W m^{-2}]
    # # See Equations 4 and 77 [1]
    # addAux(gl, 'hSo5SoOut', sensible(2*p.lambdaSo/(p.hSo5+p.hSoOut), x.tSo5, d.tSoOut))
    a.hSo5SoOut = sensible(2*p.lambdaSo/(p.hSo5+p.hSoOut), x[14], d[6])

    # # Conductive heat flux through the lumped cover [W K^{-1} m^{-2}]
    # # See comment after Equation 18 [1]
    # addAux(gl, 'hCovInCovE', sensible(\
    #     1./(p.hRf/p.lambdaRf+u.shScrPer*p.hShScrPer/p.lambdaShScrPer),\
    #     x.tCovIn, x.tCovE))
    a.hCovInCovE = sensible(\
        1/(p.hRf/p.lambdaRf + u[8]*p.hShScrPer/p.lambdaShScrPer),\
        x[5], x[6])

    # # Between lamps and air in main compartment [W m^{-2}]
    # # Equation A29 [5]
    # addAux(gl, 'hLampAir', sensible(p.cHecLampAir, x.tLamp, x.tAir))
    a.hLampAir = sensible(p.cHecLampAir, x[17], x[2])
    
    # # Between grow pipes and air in main compartment [W m^{-2}]
    # # Equations A31, A33 [5]
    # addAux(gl, 'hGroPipeAir', sensible(\
        # 1.99*pi*p.phiGroPipeE*p.lGroPipe*(abs(x.tGroPipe-x.tAir)).^0.32, \
    #     x.tGroPipe, x.tAir))
    a.hGroPipeAir = sensible(\
        1.99 * M_PI * p.phiGroPipeE * p.lGroPipe * fabs(x[19]-x[2])**0.32, \
        x[19], x[2])
        
    # # Between interlights and air in main compartment [W m^{-2}]
    # # Equation A30 [5]
    # addAux(gl, 'hIntLampAir', sensible(p.cHecIntLampAir, x.tIntLamp, x.tAir))
    a.hIntLampAir = sensible(p.cHecIntLampAir, x[18], x[2])

    # Smooth switch between day and night [-]
    # Equation 50 [1]
    # addAux(gl, 'sRs', 1./(1+exp(p.sRs.*(gl.a.rCan-p.rCanSp))))
    a.sRs = 1/(1 + exp(p.sRs*(a.rCan-p.rCanSp)))
        
    # Parameter for co2 influence on stomatal resistance [ppm{CO2}^{-2}]
    # Equation 51 [1]
    # addAux(gl, 'cEvap3', p.cEvap3Night*(1-gl.a.sRs)+p.cEvap3Day*gl.a.sRs)
    a.cEvap3 = p.cEvap3Night*(1-a.sRs) + p.cEvap3Day*a.sRs
 
    # Parameter for vapor pressure influence on stomatal resistance [Pa^{-2}]
    # addAux(gl, 'cEvap4', p.cEvap4Night*(1-gl.a.sRs)+p.cEvap4Day*gl.a.sRs)
    a.cEvap4 = p.cEvap4Night*(1-a.sRs) + p.cEvap4Day*a.sRs

    # Radiation influence on stomatal resistance [-]
    # Equation 49 [1]
    # addAux(gl, 'rfRCan', (gl.a.rCan+p.cEvap1)./(gl.a.rCan+p.cEvap2))
    a.rfRCan = (a.rCan+p.cEvap1) / (a.rCan+p.cEvap2)

    # CO2 influence on stomatal resistance [-]
    # Equation 49 [1]
    # addAux(gl, 'rfCo2', min(1.5, 1 + gl.a.cEvap3.* (p.etaMgPpm*x.co2Air-200).^2))
    a.rfCo2 = fmin(1.5, 1 + a.cEvap3 * (p.etaMgPpm*x[0] - 200)**2)
        # perhpas replace p.etaMgPpm*x.co2Air with a.co2InPpm

    # Vapor pressure influence on stomatal resistance [-]
    # Equation 49 [1]
    # addAux(gl, 'rfVp', min(5.8, 1+gl.a.cEvap4.*(satVp(x.tCan)-x.vpAir).^2))
    a.rfVp = fmin(5.8, 1 + a.cEvap4 * (satVp(x[4]) - x[15])**2)

    # Stomatal resistance [s m^{-1}]
    # Equation 48 [1]
    # addAux(gl, 'rS', p.rSMin.*gl.a.rfRCan.*gl.a.rfCo2.*gl.a.rfVp)
    a.rS = p.rSMin * a.rfRCan * a.rfCo2 * a.rfVp

    # Vapor transfer coefficient of canopy transpiration [kg m^{-2} Pa^{-1} s^{-1}]
    # Equation 47 [1]
    # addAux(gl, 'vecCanAir', 2*p.rhoAir*p.cPAir*gl.a.lai./\
    #     (p.L*p.gamma*(p.rB+gl.a.rS)))
    a.vecCanAir = 2*p.rhoAir * p.cPAir * a.lai / \
        (p.L * p.gamma * (p.rB + a.rS))

    # Canopy transpiration [kg m^{-2} s^{-1}]
    # Equation 46 [1]
    # addAux(gl, 'mvCanAir', (satVp(x.tCan)-x.vpAir).*gl.a.vecCanAir) 
    a.mvCanAir = (satVp(x[4]) - x[15]) * a.vecCanAir

    ######################
    #### Vapor Fluxes ####
    ######################

    # These are currently not used in the model..
    a.mvPadAir = 0
    a.mvFogAir = 0
    a.mvBlowAir = 0
    a.mvAirOutPad = 0

    # Condensation from main compartment on thermal screen [kg m^{-2} s^{-1}]
    # Table 4 [1], Equation 42 [1]
    # addAux(gl, 'mvAirThScr', cond(1.7*u.thScr.*nthroot(abs(x.tAir-x.tThScr),3), \
    #     x.vpAir, satVp(x.tThScr)))
    a.mvAirThScr = cond(1.7 * u[2] * fabs(x[2]-x[7])**(1/3), \
        x[15], satVp(x[7]))

    # Condensation from main compartment on blackout screen [kg m^{-2} s^{-1}]
    # Equatio A39 [5], Equation 7.39 [7]
    # addAux(gl, 'mvAirBlScr', cond(1.7*u.blScr.*nthroot(abs(x.tAir-x.tBlScr),3), \
    #     x.vpAir, satVp(x.tBlScr)))
    a.mvAirBlScr = cond(1.7 * u[7] * fabs(x[2]-x[20])**(1/3), \
        x[15], satVp(x[20]))

    # Condensation from top compartment to cover [kg m^{-2} s^{-1}]
    # Table 4 [1]
    # addAux(gl, 'mvTopCovIn', cond(p.cHecIn*nthroot(abs(x.tTop-x.tCovIn),3)*p.aCov/p.aFlr,\
    #     x.vpTop, satVp(x.tCovIn)))
    a.mvTopCovIn = cond(p.cHecIn * p.aCov/p.aFlr * fabs(x[3]-x[5])**(1/3),\
        x[16], satVp(x[5]))

    # Vapor flux from main to top compartment [kg m^{-2} s^{-1}]
    # addAux(gl, 'mvAirTop', airMv(gl.a.fScr, x.vpAir, x.vpTop, x.tAir, x.tTop))
    a.mvAirTop = airMv(a.fScr, x[15], x[16], x[2], x[3])

    # Vapor flux from top compartment to outside [kg  m^{-2} s^{-1}]
    # addAux(gl, 'mvTopOut', airMv(gl.a.fVentRoof, x.vpTop, d.vpOut, x.tTop, d.tOut))
    a.mvTopOut = airMv(a.fVentRoof, x[16], d[2], x[3], d[1])

    # Vapor flux from main compartment to outside [kg m^{-2} s^{-1}]
    # addAux(gl, 'mvAirOut', airMv(gl.a.fVentSide+gl.a.fVentForced, x.vpAir, \
    #     d.vpOut, x.tAir, d.tOut))
    a.mvAirOut = airMv(a.fVentSide+a.fVentForced, x[15], \
        d[2], x[2], d[1])

    ############################
    #### Latent heat fluxes ####
    ############################

    a.lCanAir = p.L * a.mvCanAir
    a.lAirThScr = p.L * a.mvAirThScr
    a.lAirBlScr = p.L * a.mvAirBlScr
    a.lTopCovIn = p.L * a.mvTopCovIn

    ###############################
    #### Canopy photosynthesis ####
    ###############################

    # PAR absorbed by the canopy [umol{photons} m^{-2} s^{-1}]
    # Equation 17 [2]
    # addAux(gl, 'parCan', p.zetaLampPar*gl.a.rParLampCan + p.parJtoUmolSun*gl.a.rParSunCan + \
    #     p.zetaIntLampPar*gl.a.rParIntLampCan)
    a.parCan = p.zetaLampPar*a.rParLampCan + p.parJtoUmolSun*a.rParSunCan + \
        p.zetaIntLampPar*a.rParIntLampCan

    # Maximum rate of electron transport rate at 25C [umol{e-} m^{-2} s^{-1}]
    # Equation 16 [2]
    # addAux(gl, 'j25CanMax', gl.a.lai*p.j25LeafMax)
    a.j25CanMax = a.lai*p.j25LeafMax

    # CO2 compensation point [ppm]
    # Equation 23 [2]
    # addAux(gl, 'gamma', divNoBracks(p.j25LeafMax, (gl.a.j25CanMax)*1) .*p.cGamma.*x.tCan + \
    #     20*p.cGamma.*(1-divNoBracks(p.j25LeafMax,(gl.a.j25CanMax)*1)))
    a.gamma = (p.j25LeafMax / a.j25CanMax) * p.cGamma * x[4] + \
        20 * p.cGamma * (1 - (p.j25LeafMax / a.j25CanMax))

    # CO2 concentration in the stomata [ppm]
    # Equation 21 [2]
    # addAux(gl, 'co2Stom', p.etaCo2AirStom*gl.a.co2InPpm)
    a.co2Stom = p.etaCo2AirStom * a.co2InPpm

    # # Potential rate of electron transport [umol{e-} m^{-2} s^{-1}]
    # # Equation 15 [2]
    # # Note that R in [2] is 8.314 and R in [1] is 8314
    # addAux(gl, 'jPot', gl.a.j25CanMax.*exp(p.eJ*(x.tCan+273.15-p.t25k)./(1e-3*p.R*(x.tCan+273.15)*p.t25k)).*\
    #     (1+exp((p.S*p.t25k-p.H)./(1e-3*p.R*p.t25k)))./\
    #     (1+exp((p.S*(x.tCan+273.15)-p.H)./(1e-3*p.R*(x.tCan+273.15)))))
    a.jPot = a.j25CanMax * exp(p.eJ * (x[4]+273.15-p.t25k) / (1e-3*p.R*(x[4]+273.15)*p.t25k)) * \
        (1 + exp((p.S*p.t25k-p.H) / (1e-3*p.R*p.t25k))) / \
        (1 + exp((p.S*(x[4]+273.15)-p.H) / (1e-3*p.R*(x[4]+273.15))))

    # # Electron transport rate [umol{e-} m^{-2} s^{-1}]
    # # Equation 14 [2]
    # addAux(gl, 'j', (1/(2*p.theta))*(gl.a.jPot+p.alpha*gl.a.parCan-\
    #     sqrt((gl.a.jPot+p.alpha*gl.a.parCan).^2-4*p.theta*gl.a.jPot.*p.alpha.*gl.a.parCan)))
    a.j = (1/(2*p.theta)) * (a.jPot+p.alpha*a.parCan -\
        sqrt((a.jPot+p.alpha*a.parCan)**2 - 4*p.theta*a.jPot*p.alpha*a.parCan))

    # # Photosynthesis rate at canopy level [umol{co2} m^{-2} s^{-1}]
    # # Equation 12 [2]
    # addAux(gl, 'p', gl.a.j.*(gl.a.co2Stom-gl.a.gamma)./(4*(gl.a.co2Stom+2*gl.a.gamma)))
    a.p = a.j*(a.co2Stom-a.gamma) / (4*(a.co2Stom + 2*a.gamma))

    # # Photrespiration [umol{co2} m^{-2} s^{-1}]
    # # Equation 13 [2]
    # addAux(gl, 'r', gl.a.p.*gl.a.gamma./gl.a.co2Stom)
    a.r = a.p*a.gamma / a.co2Stom

    # # Inhibition due to full carbohydrates buffer [-]
    # # Equation 11, Equation B.1, Table 5 [2]
    # addAux(gl, 'hAirBuf', 1./(1+exp(5e-4*(x.cBuf-p.cBufMax))))
    a.hAirBuf = 1/(1+ exp(5e-4*(x[22]-p.cBufMax)))

    # # Net photosynthesis [mg{CH2O} m^{-2} s^{-1}]
    # # Equation 10 [2]
    # addAux(gl, 'mcAirBuf', p.mCh2o*gl.a.hAirBuf.*(gl.a.p-gl.a.r))
    a.mcAirBuf = p.mCh2o * a.hAirBuf * (a.p - a.r)

    # ## Carbohydrate buffer
    # # Temperature effect on structural carbon flow to organs
    # # Equation 28 [2]
    # addAux(gl, 'gTCan24', 0.047*x.tCan24+0.06)
    a.gTCan24 = 0.047*x[21] + 0.06

    # # Inhibition of carbohydrate flow to the organs
    # # Equation B.3 [2]
    # addAux(gl, 'hTCan24', 1./(1+exp(-1.1587*(x.tCan24-p.tCan24Min))).* \
    #     1./(1+exp(1.3904*(x.tCan24-p.tCan24Max))))
    a.hTCan24 = 1 / (1 + exp(-1.1587*(x[21]-p.tCan24Min))) * \
        1 / (1 + exp(1.3904*(x[21]-p.tCan24Max)))

    # # Inhibition of carbohydrate flow to the fruit
    # # Equation B.3 [2]
    # addAux(gl, 'hTCan', 1./(1+exp(-0.869*(x.tCan-p.tCanMin))).* \
    #     1./(1+exp(0.5793*(x.tCan-p.tCanMax))))
    a.hTCan = 1 / (1 + exp(-0.869*(x[4]-p.tCanMin))) * \
        1 / (1 + exp(0.5793*(x[4]-p.tCanMax)))

    # # Inhibition due to development stage 
    # # Equation B.6 [2]
    # gl, 'hTCanSum', 0.5*(x.tCanSum/p.tEndSum+\
    #     sqrt((x.tCanSum./p.tEndSum).^2+1e-4)) - \
    #     0.5*((x.tCanSum-p.tEndSum)./p.tEndSum+\
    #     sqrt(((x.tCanSum-p.tEndSum)/p.tEndSum).^2 + 1e-4))
    a.hTCanSum = 0.5 *(x[26] / p.tEndSum + \
        sqrt((x[26] / p.tEndSum)**2 + 1e-4)) - \
        0.5 * ((x[26] - p.tEndSum) / p.tEndSum + \
        sqrt(((x[26] - p.tEndSum) / p.tEndSum)**2 + 1e-4))

    # # Inhibition due to insufficient carbohydrates in the buffer [-]
    # # Equation 26 [2]
    # gl, 'hBufOrg', 1./(1+exp(-5e-3*(x.cBuf-p.cBufMin)))
    a.hBufOrg = 1 / (1 + exp(-5e-3*(x[22] - p.cBufMin)))

    # # Carboyhdrate flow from buffer to leaves [mg{CH2O} m^{2} s^{-1}]
    # Equation 25 [2]
    # addAux(gl, 'mcBufLeaf', gl.a.hBufOrg.*gl.a.hTCan24.*gl.a.gTCan24.*gl.p.rgLeaf)
    a.mcBufLeaf = a.hBufOrg * a.hTCan24 * a.gTCan24 * p.rgLeaf

    # # Carboyhdrate flow from buffer to stem [mg{CH2O} m^{2} s^{-1}]
    # # Equation 25 [2]
    # addAux(gl, 'mcBufStem', gl.a.hBufOrg.*gl.a.hTCan24.*gl.a.gTCan24.*gl.p.rgStem)
    a.mcBufStem = a.hBufOrg * a.hTCan24 * a.gTCan24 * p.rgStem

    # # Carboyhdrate flow from buffer to fruit [mg{CH2O} m^{2} s^{-1}]
    # # Equation 24 [2]
    # addAux(gl, 'mcBufFruit', gl.a.hBufOrg.*\
    #     gl.a.hTCan.*gl.a.hTCan24.*gl.a.hTCanSum.*gl.a.gTCan24.*gl.p.rgFruit)
    a.mcBufFruit = a.hBufOrg * a.hTCan * a.hTCan24 * a.hTCanSum * a.gTCan24 * p.rgFruit

    # Growth respiration [mg{CH2O} m^{-2] s^{-1}]
    # Equations 43-44 [2]
    # addAux(gl, 'mcBufAir', p.cLeafG*gl.a.mcBufLeaf + p.cStemG*gl.a.mcBufStem \
    #     +p.cFruitG*gl.a.mcBufFruit)
    a.mcBufAir = p.cLeafG*a.mcBufLeaf + p.cStemG*a.mcBufStem + p.cFruitG*a.mcBufFruit

    # Leaf maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    # Equation 45 [2]
    # addAux(gl, 'mcLeafAir', (1-exp(-p.cRgr*p.rgr)).*p.q10m.^(0.1*(x.tCan24-25)).* \
    #     x.cLeaf*p.cLeafM)
    a.mcLeafAir = (1- exp(-p.cRgr*p.rgr)) * p.q10m**(0.1*(x[21]-25)) * \
        x[23] * p.cLeafM

    # Stem maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    # Equation 45 [2]
    # addAux(gl, 'mcStemAir', (1-exp(-p.cRgr*p.rgr)).*p.q10m.^(0.1*(x.tCan24-25)).* \
    #     x.cStem*p.cStemM)
    a.mcStemAir = (1- exp(-p.cRgr*p.rgr)) * p.q10m**(0.1*(x[21]-25)) * \
        x[24] * p.cStemM

    # Fruit maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    # Equation 45 [2]
    # addAux(gl, 'mcFruitAir', (1-exp(-p.cRgr*p.rgr)).*p.q10m.^(0.1*(x.tCan24-25)).* \
    #     x.cFruit*p.cFruitM)
    a.mcFruitAir = (1- exp(-p.cRgr*p.rgr)) * p.q10m**(0.1*(x[21]-25)) * \
        x[25] * p.cFruitM

    # Total maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    # Equation 45 [2]
    # addAux(gl, 'mcOrgAir', gl.a.mcLeafAir+gl.a.mcStemAir+gl.a.mcFruitAir)
    a.mcOrgAir = a.mcLeafAir + a.mcStemAir + a.mcFruitAir

    ## Leaf pruning and fruit harvest
    # A new smoothing function has been applied here to avoid stiffness
    # Leaf pruning [mg{CH2O} m^{-2] s^{-1}]
    # Equation B.5 [2]
    # addAux(gl, 'mcLeafHar', smoothHar(x.cLeaf, p.cLeafMax, 1e4, 5e4))
    a.mcLeafHar = smoothHar(x[23], p.cLeafMax, 1e4, 5e4)

    # Fruit harvest [mg{CH2O} m^{-2} s^{-1}]
    # Equation A45 [5], Equation 7.45 [7]
    # addAux(gl, 'mcFruitHar', smoothHar(x.cFruit, p.cFruitMax, 1e4, 5e4))
    a.mcFruitHar = smoothHar(x[25], p.cFruitMax, 1e4, 5e4)
    # print("", mcFruitHar)

    # comptures sum of the harvest fruit over the past timestep
    a.mcFruitHarSum += a.mcFruitHar

    # Net crop assimilation [mg{CO2} m^{-2} s^{-1}]
    # It is assumed that for every mol of CH2O in net assimilation, a mol
    # of CO2 is taken from the air, thus the conversion uses molar masses
    # addAux(gl, 'mcAirCan', (p.mCo2/p.mCh2o)*(gl.a.mcAirBuf-gl.a.mcBufAir-gl.a.mcOrgAir))
    a.mcAirCan = (p.mCo2/p.mCh2o) * (a.mcAirBuf-a.mcBufAir-a.mcOrgAir)
    # Other CO2 flows [mg{CO2} m^{-2} s^{-1}]
    # Equation 45 [1]

    # From main to top compartment 
    # addAux(gl, 'mcAirTop', airMc(gl.a.fScr, x.co2Air, x.co2Top))
    a.mcAirTop = airMc(a.fScr, x[0], x[1])

    # From top compartment outside
    # addAux(gl, 'mcTopOut', airMc(gl.a.fVentRoof, x.co2Top, d.co2Out))
    a.mcTopOut = airMc(a.fVentRoof, x[1], d[3])

    # From main compartment outside
    # addAux(gl, 'mcAirOut', airMc(gl.a.fVentSide+gl.a.fVentForced, x.co2Air, d.co2Out))
    a.mcAirOut = airMc(a.fVentSide+a.fVentForced, x[0], d[3])

    ## Heat from boiler - Section 9.2 [1]

    # Heat from boiler to pipe rails [W m^{-2}]
    # Equation 55 [1]
    # addAux(gl, 'hBoilPipe', u.boil*p.pBoil/p.aFlr)
    a.hBoilPipe = u[0] * p.pBoil / p.aFlr

    # Heat from boiler to grow pipes [W m^{-2}]
    # addAux(gl, 'hBoilGroPipe', u.boilGro*p.pBoilGro/p.aFlr)
    a.hBoilGroPipe = u[6] * p.pBoilGro / p.aFlr

    ## External CO2 source - Section 9.9 [1]

    # CO2 injection [mg m^{-2} s^{-1}]
    # Equation 76 [1]
    # addAux(gl, 'mcExtAir', u.extCo2*p.phiExtCo2/p.aFlr)
    a.mcExtAir = u[1] * p.phiExtCo2 / p.aFlr

    ## Objects not currently included in the model
    # addAux(gl, 'mcBlowAir', DynamicElement('0',0))
    # addAux(gl, 'mcPadAir', DynamicElement('0',0))
    # addAux(gl, 'hPadAir', DynamicElement('0',0))
    # addAux(gl, 'hPasAir', DynamicElement('0',0))
    # addAux(gl, 'hBlowAir', DynamicElement('0',0))
    # addAux(gl, 'hAirPadOut', DynamicElement('0',0))
    # addAux(gl, 'hAirOutPad', DynamicElement('0',0))
    # addAux(gl, 'lAirFog', DynamicElement('0',0))
    # addAux(gl, 'hIndPipe', DynamicElement('0',0))
    # addAux(gl, 'hGeoPipe', DynamicElement('0',0))

    a.mcBlowAir = 0
    a.mcPadAir = 0
    a.hPadAir = 0
    a.hPasAir = 0
    a.hBlowAir = 0
    a.hAirPadOut = 0
    a.hAirOutPad = 0
    a.lAirFog = 0
    a.hIndPipe = 0
    a.hGeoPipe = 0

    ## Lamp cooling
	# Equation A34 [5], Equation 7.34 [7]
    # addAux(gl, 'hLampCool', p.etaLampCool*gl.a.qLampIn)
    a.hLampCool = p.etaLampCool * a.qLampIn
    
    ## Heat harvesting, mechanical cooling and dehumidification
    # By default there is no mechanical cooling or heat harvesting
    # see addHeatHarvesting.m for mechanical cooling and heat harvesting
    # addAux(gl, 'hecMechAir', '0')
    # addAux(gl, 'hAirMech', '0')
    # addAux(gl, 'mvAirMech', '0')
    # addAux(gl, 'lAirMech', '0')
    # addAux(gl, 'hBufHotPipe', '0')

    a.hecMechAir = 0
    a.hAirMech = 0
    a.mvAirMech = 0
    a.lAirMech = 0
    a.hBufHotPipe = 0