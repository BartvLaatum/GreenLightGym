from libc.math cimport exp, INFINITY, pi

cdef packed struct Parameters:
    char alfaLeafAir        # Convective heat transfer coefficient between leaf and greenhouse air
    double L                # Latent heat of evaporation
    double sigma            # Stefan-Boltzmann constant
    char epsCan             # FIR emission coefficient of canopy
    char epsSky             # FIR emission coefficient of the sky
    float etaGlobNir        # Ratio of NIR in global radiation
    float etaGlobPar        # Ratio of PAR in global radiation
    
    double etaMgPpm         # CO2 conversion factor from mg/m^{3} to ppm
    double etaRoofThr       # Ratio between roof vent area and total vent area where no chimney effects is assumed
    double rhoAir0          # Density of air at sealevel
    double rhoCanPar        # Density of PAR
    double rhoCanNir        # Density of NIR
    short rhoSteel          # Density of steel
    short rhoWater          # Density of water
    double gamma            # Psychrometric constant
    double omega            # Yearly frequency to calculate soil temperature
    short capLeaf           # Heat capacity of canopy leaves
    double cEvap1           # Coefficient for radiation effect on stomatal resistance
    double cEvap2           # Coeficient for radiation effect on stomatal resistance

    double cEvap3Day        # Coefficient for co2 effect on stomatal resistance (day)
    double cEvap3Night      # Coefficient for co2 effect on stomatal resistance (night)
    double cEvap4Day        # Coefficient for vapor pressure effect on stomatal resistance (day)
    double cEvap4Night      # Coefficient for vapor pressure effect on stomatal resistance (night)
    short cPAir             # Specific heat capacity of air
    short cPSteel           # Specific heat capacity of steel
    short cPWater           # Specific heat capacity of water
    double g                # Acceleration of gravity
    double hSo1             # Thickness of soil layer 1
    double hSo2             # Thickness of soil layer 2
    double hSo3             # Thickness of soil layer 3
    double hSo4             # Thickness of soil layer 4
    double hSo5             # Thickness of soil layer 5
    double k1Par            # PAR extinction coefficient of the canopy
    double k2Par            # PAR extinction coefficient of the canopy for light reflected from the floor
    double kNir             # NIR extinction coefficient of the canopy
    double kFir             # FIR extinction coefficient of the canopy
    double mAir             # Molar mass of air
    double hSoOut           # Thickness of the external soil layer

    char mWater             # Molar mass of water
    short R                 # Molar gas constant
    char rCanSp             # Radiation value above the canopy when night becomes day
    short rB                # Boundary layer resistance of the canopy for transpiration
    char rSMin              # Minimum canopy resistance for transpiration
    char sRs                # Slope of smoothed stomatal resistance model

    double etaGlobAir       # Ratio of global radiation absorbed by the greenhouse construction
    char psi                # Mean greenhouse cover slope
    short aFlr              # Floor area of greenhouse
    short aCov              # Surface of the cover including side walls
    double hAir             # Height of the main compartment
    double hGh              # Mean height of the greenhouse
    double cHecIn           # Convective heat exchange between cover and outdoor air
    double cHecOut1         # Convective heat exchange parameter between cover and outdoor air
    double cHecOut2         # Convective heat exchange parameter between cover and outdoor air
    char cHecOut3           # Convective heat exchange parameter between cover and outdoor air
    char hElevation         # Altitude of greenhouse

    short aRoof             # Roof ventilation area
    double hVent             # Vertical dimension of single ventilation opening
    char etaInsScr          # Porosity of the insect screen
    char aSide              # Side ventilation area
    float cDgh              # Ventilation discharge coefficient
    double cLeakage          # Greenhouse leakage coefficient
    double cWgh              # Ventilation global wind pressure coefficient
    char hSideRoof          # Vertical distance between mid points of side wall and roof ventilation opening

    double epsRfFir          # FIR emission coefficient of the roof
    short rhoRf             # Density of the roof layer
    double rhoRfNir          # NIR reflection coefficient of the roof
    double rhoRfPar          # PAR reflection coefficient of the roof
    double rhoRfFir          # FIR reflection coefficient of the roof
    double tauRfNir          # NIR transmission coefficient of the roof
    double tauRfPar          # PAR transmission coefficient of the roof
    double tauRfFir          # FIR transmission coefficient of the roof
    double lambdaRf          # Thermal heat conductivity of the roof
    short cPRf              # Specific heat capacity of roof layer
    double hRf               # Thickness of roof layer

    # char epsPerFir      # FIR emission coefficient of the whitewash
    # char rhoShScrPer         # Density of the whitewash
    # char rhoShScrPerNir      # NIR reflection coefficient of whitewash
    # char rhoShScrPerPar      # PAR reflection coefficient of whitewash
    # char rhoShScrPerFir      # FIR reflection coefficient of whitewash
    # char tauShScrPerNir      # NIR transmission coefficient of whitewash
    # char tauShScrPerPar      # PAR transmission coefficient of whitewash
    # char tauShScrPerFir      # FIR transmission coefficient of whitewash
    # double lambdaShScrPer    # Thermal heat conductivity of the whitewash
    # char cPShScrPer          # Specific heat capacity of the whitewash
    # char hShScrPer           # Thickness of the whitewash

    # char rhoShScrShScrNir         # NIR reflection coefficient of shadow screen
    # char rhoShScrPar         # PAR reflection coefficient of shadow screen
    # char rhoShScrFir         # FIR reflection coefficient of shadow screen
    # char tauShScrNir         # NIR transmission coefficient of shadow screen
    # char tauShScrPar         # PAR transmission coefficient of shadow screen
    # char tauShScrFir         # FIR transmission coefficient of shadow screen
    # char etaShScrCd          # Effect of shadow screen on discharge coefficient
    # char etaShScrCw          # Effect of shadow screen on wind pressure coefficient
    # char kShScr              # Shadow screen flux coefficient

    double epsThScrFir       # FIR emission coefficient of the thermal screen
    unsigned char rhoThScr   # Density of thermal screen
    double rhoThScrNir       # NIR reflection coefficient of thermal screen
    double rhoThScrPar       # PAR reflection coefficient of thermal screen
    double rhoThScrFir       # FIR reflection coefficient of thermal screen
    double tauThScrNir       # NIR transmission coefficient of thermal screen
    double tauThScrPar       # PAR transmission coefficient of thermal screen
    double tauThScrFir       # FIR transmission coefficient of thermal screen
    short cPThScr            # Specific heat capacity of thermal screen
    double hThScr            # Thickness of thermal screen
    double kThScr            # Thermal screen flux coefficient

    double epsBlScrFir       # FIR emission coefficient of the blackout screen
    unsigned char rhoBlScr   # Density of blackout screen
    double rhoBlScrNir       # NIR reflection coefficient of blackout screen
    double rhoBlScrPar       # PAR reflection coefficient of blackout screen
    double tauBlScrNir       # NIR transmission coefficient of blackout screen
    double tauBlScrPar       # PAR transmission coefficient of blackout screen
    double tauBlScrFir       # FIR transmission coefficient of blackout screen
    short cPBlScr            # Specific heat capacity of blackout screen
    double hBlScr            # Thickness of blackout screen
    double kBlScr            # Blackout screen flux coefficient

    char epsFlr             # FIR emission coefficient of the floor
    short rhoFlr            # Density of the floor
    float rhoFlrNir         # NIR reflection coefficient of the floor
    double rhoFlrPar        # PAR reflection coefficient of the floor
    double lambdaFlr        # Thermal heat conductivity of the floor
    short cPFlr             # Specific heat capacity of the floor
    double hFlr             # Thickness of floor

    int rhoCpSo             # Volumetric heat capacity of the soil
    double lambdaSo         # Thermal heat conductivity of the soil layers

    double epsPipe          # FIR emission coefficient of the heating pipes
    double phiPipeE         # External diameter of pipes
    double phiPipeI         # Internal diameter of pipes
    double lPipe            # Length of heating pipes per gh floor area
    int pBoil               # Capacity of the heating system

    int phiExtCo2           # Capacity of external CO2 source
    double capPipe          # Heat capacity of heating pipes
    double rhoAir           # Density of air

    double capAir           # Heat capacity of air
    double capFlr           # Heat capacity of floor
    double capSo1           # Heat capacity of soil layer 1
    double capSo2           # Heat capacity of soil layer 2
    double capSo3           # Heat capacity of soil layer 3
    double capSo4           # Heat capacity of soil layer 4
    double capSo5           # Heat capacity of soil layer 5
    double capThScr         # Heat capacity of thermal screen
    double capTop           # Heat capacity of air in top compartments
    double capBlScr         # Heat capacity of blackout screen

    double capCo2Air        # Capacity for CO2 in air
    double capCo2Top        # Capacity for CO2 in top compartments

    double aPipe            # Surface of pipes for floor area
    double fCanFlr          # View factor from canopy to floor
    double pressure         # Absolute air pressure at given elevation

    double globJtoUmol          # Conversion factor from global radiation to PAR
    unsigned char j25LeafMax    # Maximal rate of electron transport at 25�C of the leaf
    double cGamma               # Effect of canopy temperature on CO2 compensation point
    double etaCo2AirStom        # Conversion from greenhouse air co2 concentration and stomatal co2 concentration
    unsigned short eJ           # Activation energy for Jpot calcualtion
    double t25k                 # Reference temperature for Jpot calculation
    short S                     # Enthropy term for Jpot calculation
    int H                       # Deactivation energy for Jpot calculation
    double theta                # Degree of curvature of the electron transport rate
    double alpha                # Conversion factor from photons to electrons including efficiency term
    double mCh2o                # Molar mass of CH2O
    double mCo2                 # Molar mass of CO2

    double parJtoUmolSun        # Conversion factor of sun's PAR from J to umol{photons} J^{-1}
    char laiMax                 # Max leaf area index
    double sla                  # Specific leaf area
    double rgr                  # Relative growth rate
    double cLeafMax             # Maximum leaf size

    int cFruitMax               # Maximum fruit size
    double cFruitG              # Fruit growth respiration coefficient
    double cLeafG               # Leaf growth respiration coefficient
    double cStemG               # Stem growth respiration coefficient
    int cRgr                    # Regression coefficient in maintenance respiration function
    char q10m                   # Q10 value of temperature effect on maintenance respiration
    double cFruitM              # Fruit maintenance respiration coefficient
    double cLeafM               # Leaf maintenance respiration coefficient
    double cStemM               # Stem maintenance respiration coefficient
    
    double rgFruit              # Potential fruit growth coefficient
    double rgLeaf               # Potential leaf growth coefficient
    double rgStem               # Potential stem growth coefficient

    short cBufMax           # Maximum capacity of carbohydrate buffer
    short cBufMin           # Minimum capacity of carbohydrate buffer
    double tCan24Max         # Inhibition of carbohydrate flow because of high temperatures
    char tCan24Min          # Inhibition of carbohydrate flow because of low temperatures
    char tCanMax            # Inhibition of carbohydrate flow because of high instantenous temperatures
    char tCanMin            # Inhibition of carbohydrate flow because of low instantenous temperatures
    short tEndSum           # Temperature sum where crop is fully generative

    char rhMax              # Upper bound on relative humidity
    char dayThresh          # Threshold to consider switch from night to day
    float tSpDay           # Heat is on below this point in day
    double tSpNight         # Heat is on below this point in night
    char tHeatBand        # P-band for heating
    char tVentOff         # Distance from heating setpoint where ventilation stops (even if humidity is too high)
    char tScreenOn        # Distance from screen setpoint where screen is on (even if humidity is too high)
    char thScrSpDay       # Screen is closed at day when outdoor is below this temperature
    char thScrSpNight     # Screen is closed at night when outdoor is below this temperature
    char thScrPband       # P-band for thermal screen
    short co2SpNight
    short co2SpDay         # Co2 is supplied if co2 is below this point during day
    char co2Band          # P-band for co2 supply
    char heatDeadZone     # Zone between heating setpoint and ventilation setpoint
    char ventHeatPband    # P-band for ventilation due to excess heat
    char ventColdPband    # P-band for ventilation due to low indoor temperature
    char ventRhPband      # P-band for ventilation due to relative humidity
    char thScrRh          # Relative humidity where thermal screen is forced to open, with respect to rhMax
    char thScrRhPband     # P-band for thermal screen opening due to excess relative humidity
    char thScrDeadZone    # Zone between heating setpoint and point where screen opens

    char lampsOn          # Time of day to switch on lamps
    char lampsOff         # Time of day to switch off lamps

    char dayLampStart     # Day of year when lamps start
    short dayLampStop      # Day of year when lamps stop

    short lampsOffSun      # Lamps are switched off if global radiation is above this value
    char lampRadSumLimit  # Predicted daily radiation sum from the sun where lamps are not used that day
    char lampExtraHeat    # Control for lamps due to too much heat - switched off if indoor temperature is above setpoint+heatDeadZone+lampExtraHeat
    char blScrExtraRh     # Control for blackout screen due to humidity - screens open if relative humidity exceeds rhMax+blScrExtraRh
    char useBlScr         # Determines whether a blackout screen is used (1 if used, 0 otherwise)

    char mechCoolPband    # P-band for mechanical cooling
    char mechDehumidPband # P-band for mechanical dehumidification
    char heatBufPband     # P-band for heating from the buffer
    char mechCoolDeadZone # Zone between heating setpoint and mechanical cooling setpoint

    char epsGroPipe       # Emissivity of grow pipes
    double lGroPipe         # Length of grow pipes per gh floor area
    double phiGroPipeE      # External diameter of grow pipes
    double phiGroPipeI      # Internal diameter of grow pipes

    double aGroPipe         # Surface area of pipes for floor area
    char pBoilGro         # Capacity of the grow pipe heating system
    double capGroPipe       # Heat capacity of grow pipes

    double thetaLampMax     # Maximum intensity of lamps
    char heatCorrection   # correction for temperature setpoint when lamps are on
    double etaLampPar       # fraction of lamp input converted to PAR
    double etaLampNir       # fraction of lamp input converted to NIR
    double tauLampPar       # transmissivity of lamp layer to PAR
    char rhoLampPar       # reflectivity of lamp layer to PAR
    double tauLampNir       # transmissivity of lamp layer to NIR
    char rhoLampNir       # reflectivity of lamp later to NIR
    double tauLampFir       # transmissivity of lamp later to FIR
    double aLamp            # lamp area
    double epsLampTop       # emissivity of top side of lamp
    double epsLampBottom    # emissivity of bottom side of lamp
    short capLamp          # heat capacity of lamp
    double cHecLampAir      # heat exchange coefficient of lamp
    char etaLampCool      # fraction of lamp input removed by cooling
    double zetaLampPar      # J to umol conversion of PAR output of lamp

    char intLamps           # whether we use intercropping lamps
    float vIntLampPos     # Vertical position of the interlights within the canopy [0-1, 0 is above canopy and 1 is below]
    float fIntLampDown    # Fraction of interlight light output of lamps that is directed downwards
    char capIntLamp       # Capacity of interlight lamps
    char etaIntLampPar    # fraction of interlight lamp input converted to PAR
    char etaIntLampNir    # fraction of interlight lamp input converted to NIR
    char aIntLamp         # interlight lamp area
    char epsIntLamp      # emissivity of interlight lamp
    char thetaIntLampMax  # Maximum intensity of interlight lamps
    char zetaIntLampPar   # J to umol conversion of PAR output of interlight lamp
    char cHecIntLampAir   # heat exchange coefficient of interlight lamp
    char tauIntLampFir    # transmissivity of interlight lamp later to FIR
    double k1IntPar           # PAR extinction coefficient of the canopy
    double k2IntPar           # PAR extinction coefficient of the canopy for light reflected from the floor
    double kIntNir            # NIR extinction coefficient of the canopy
    double kIntFir            # FIR extinction coefficient of the canopy

    float cLeakTop          # Fraction of leakage ventilation going from the top 
    double minWind          #  wind speed where the effect of wind on leakage begins

# Initialize the values of a Parameters struct
cdef inline void initParameters(Parameters* p, char noLamps, char ledLamps, char hpsLamps, char intLamps):
    p.alfaLeafAir = 5
    p.L = 2.45e6
    p.sigma = 5.67e-8
    p.epsCan = 1
    p.epsSky = 1
    p.etaGlobNir = 0.5
    p.etaGlobPar = 0.5

    p.etaMgPpm = 0.554
    p.etaRoofThr = 0.9
    p.rhoAir0 = 1.2 
    p.rhoCanPar = 0.07
    p.rhoCanNir = 0.35
    p.rhoSteel = 7850
    p.rhoWater = 1000
    p.gamma = 65.8
    p.omega = 1.99e-7
    p.capLeaf = 1200
    p.cEvap1 = 4.3
    p.cEvap2 = 0.54

    p.cEvap3Day = 6.1e-7
    p.cEvap3Night = 1.1e-11
    p.cEvap4Day = 4.3e-6
    p.cEvap4Night = 5.2e-6
    p.cPAir = 1000
    p.cPSteel = 640
    p.cPWater = 4180
    p.g = 9.81
    p.hSo1 = 0.04 
    p.hSo2 = 0.08 
    p.hSo3 = 0.16 
    p.hSo4 = 0.32 
    p.hSo5 = 0.64 
    p.k1Par = 0.7 
    p.k2Par = 0.7 
    p.kNir = 0.27 
    p.kFir = 0.94 
    p.mAir = 28.96
    p.hSoOut = 1.28

    p.mWater = 18
    p.R = 8314
    p.rCanSp = 5
    p.rB = 275
    p.rSMin = 82
    p.sRs = -1

    ## Contstruction
    p.etaGlobAir = 0.1
    p.psi = 25
    p.aFlr = 14000
    p.aCov = 18000
    p.hAir = 3.8
    p.hGh = 4.2
    p.cHecIn = 1.86
    p.cHecOut1 = 2.8
    p.cHecOut2 = 1.2
    p.cHecOut3 = 1
    p.hElevation = 0

    ## Ventialtion
    p.aRoof = 1400
    p.hVent = 0.68
    p.etaInsScr = 1
    p.aSide = 0
    p.cDgh = 0.75
    p.cLeakage = 1e-4
    p.cWgh = 0.09
    p.hSideRoof = 0

    ## Roof parameters
    p.epsRfFir = 0.85
    p.rhoRf = 2600
    p.rhoRfNir = 0.13
    p.rhoRfPar = 0.13
    p.rhoRfFir = 0.15
    p.tauRfNir = 0.85
    p.tauRfPar = 0.85
    p.tauRfFir = 0
    p.lambdaRf = 1.05
    p.cPRf = 840
    p.hRf = 4e-3

    ## Whitewash
    # p.epsShScrPerFir = 0
    # p.rhoShScrPer = 0
    # p.rhoShScrPerNir = 0
    # p.rhoShScrPerPar = 0
    # p.rhoShScrPerFir = 0
    # p.tauShScrPerNir = 1
    # p.tauShScrPerPar = 1
    # p.tauShScrPerFir = 1
    # p.lambdaShScrPer = INFINITY
    # p.cPShScrPer = 0
    # p.hShScrPer = 0

    ## Shadow screen 
    # p.rhoShScrNir = 0
    # p.rhoShScrPar = 0
    # p.rhoShScrFir = 0
    # p.tauShScrNir = 1
    # p.tauShScrPar = 1
    # p.tauShScrFir = 1
    # p.etaShScrCd = 0
    # p.etaShScrCw = 0
    # p.kShScr = 0

    ## Thermal Screen
    p.epsThScrFir = 0.67
    p.rhoThScr = 200 	
    p.rhoThScrNir = 0.35
    p.rhoThScrPar = 0.35
    p.rhoThScrFir = 0.18
    p.tauThScrNir = 0.6 
    p.tauThScrPar = 0.6 
    p.tauThScrFir = 0.15
    p.cPThScr = 1800 	
    p.hThScr = 0.35e-3 
    p.kThScr = 0.05e-3 

    ## Blackout screen
    p.epsBlScrFir = 0.67
    p.rhoBlScr = 200
    p.rhoBlScrNir = 0.35
    p.rhoBlScrPar = 0.35
    p.tauBlScrNir = 0.01
    p.tauBlScrPar = 0.01
    p.tauBlScrFir = 0.7
    p.cPBlScr = 1800
    p.hBlScr = 0.35e-3
    p.kBlScr = 0.05e-3

    ## Floor
    p.epsFlr = 1
    p.rhoFlr = 2300
    p.rhoFlrNir = 0.5
    p.rhoFlrPar = 0.65
    p.lambdaFlr = 1.7
    p.cPFlr = 880
    p.hFlr = 0.02

    ## Soil
    p.rhoCpSo = 1730000
    p.lambdaSo = 0.85

    ## Heating system
    p.epsPipe = 0.88
    p.phiPipeE = 51e-3
    p.phiPipeI = 47e-3
    p.lPipe = 1.875
    p.pBoil = 130*p.aFlr

    ## Active climate control [1]
    p.phiExtCo2 = 72000             # Capacity of external CO2 source [mg s^{-1}]

    ## Heat capacity of heating pipes [J K^{-1} m^{-2}]
    p.capPipe = 0.25*pi * p.lPipe*(( p.phiPipeE**2- p.phiPipeI**2)* p.rhoSteel*\
          p.cPSteel+ p.phiPipeI**2* p.rhoWater* p.cPWater)

    ## Density of air [kg m^{-3}]
    p.rhoAir =  p.rhoAir0*exp(p.g* p.mAir* p.hElevation/(293.15* p.R))

    # Heat capacity of greenhouse objects [J K^{-1} m^{-2}]
    p.capAir =  p.hAir * p.rhoAir * p.cPAir
    p.capFlr =  p.hFlr * p.rhoFlr* p.cPFlr 
    p.capSo1 =  p.hSo1 * p.rhoCpSo
    p.capSo2 =  p.hSo2 * p.rhoCpSo
    p.capSo3 =  p.hSo3 * p.rhoCpSo
    p.capSo4 =  p.hSo4 * p.rhoCpSo
    p.capSo5 =  p.hSo5 * p.rhoCpSo
    p.capThScr =  p.hThScr * p.rhoThScr * p.cPThScr
    p.capTop = ( p.hGh - p.hAir) * p.rhoAir * p.cPAir
    p.capBlScr =  p.hBlScr * p.rhoBlScr * p.cPBlScr

    # Capacity for CO2 [m]
    p.capCo2Air = p.hAir
    p.capCo2Top = p.hGh- p.hAir

    # Surface of pipes for floor area [-]
    # Table 3 [1]
    p.aPipe = pi * p.lPipe * p.phiPipeE

    # View factor from canopy to floor
    # Table 3 [1]
    p.fCanFlr = 1 - 0.49*pi * p.lPipe * p.phiPipeE

    # Absolute air pressure at given elevation [Pa]
    # See https://www.engineeringtoolbox.com/air-altitude-pressure-d_462.html
    p.pressure = 101325*(1 - 2.5577e-5*p.hElevation)**5.25588

    ## Canopy photosynthesis
    p.globJtoUmol = 2.3
    p.j25LeafMax = 210
    p.cGamma = 1.7
    p.etaCo2AirStom = 0.67
    p.eJ = 37000
    p.t25k = 298.15
    p.S = 710
    p.H = 220000
    p.theta = 0.7
    p.alpha = 0.385
    p.mCh2o = 30e-3
    p.mCo2 = 44e-3

    p.parJtoUmolSun = 4.6 

    p.laiMax = 3
    p.sla = 2.66e-5
    p.rgr = 3e-6
    p.cLeafMax = p.laiMax/ p.sla

    p.cFruitMax = 300_000

    p.cFruitG = 0.27
    p.cLeafG = 0.28
    p.cStemG = 0.3
    p.cRgr = 2_850_000
    p.q10m = 2
    p.cFruitM = 1.16e-7
    p.cLeafM = 3.47e-7
    p.cStemM = 1.47e-7

    p.rgFruit = 0.328	
    p.rgLeaf = 0.095
    p.rgStem = 0.074

    ## Carbohydrates buffer
    p.cBufMax = 20_000
    p.cBufMin = 1000
    p.tCan24Max = 24.5
    p.tCan24Min = 15
    p.tCanMax = 34
    p.tCanMin = 10

    ## Crop development
    p.tEndSum = 1035

    ## Control parameters
    p.rhMax = 90
    p.dayThresh = 20
    p.tSpDay = 19.5
    p.tSpNight = 16.5
    p.tHeatBand = -1
    p.tVentOff = 1
    p.tScreenOn = 2
    p.thScrSpDay = 5
    p.thScrSpNight = 10
    p.thScrPband = -1
    p.co2SpNight = 500
    p.co2SpDay = 800
    p.co2Band = -100
    p.heatDeadZone = 5
    p.ventHeatPband = 4
    p.ventColdPband = -1
    p.ventRhPband = 5
    p.thScrRh = -2
    p.thScrRhPband = 2
    p.thScrDeadZone = 4

    p.lampsOn = 0          
    p.lampsOff = 0         

    p.dayLampStart = -1
    p.dayLampStop = 400

    p.lampsOffSun = 400 
    p.lampRadSumLimit = 10
    p.lampExtraHeat = 2
    p.blScrExtraRh = 100
    p.useBlScr = 0

    p.mechCoolPband = 1
    p.mechDehumidPband = 2
    p.heatBufPband = -1
    p.mechCoolDeadZone = 2

    ## Grow pipe parameters
    p.epsGroPipe = 0

    # There are no grow pipes so these parameters are not important, but
    # they cannot be 0 because the ODE for the grow pipe still exists
    p.lGroPipe = 1.655
    p.phiGroPipeE = 35e-3
    p.phiGroPipeI = (35e-3)-(1.2e-3)
        
    p.aGroPipe = pi* p.lGroPipe* p.phiGroPipeE
    p.pBoilGro = 0

    # Heat capacity of grow pipes [J K^{-1} m^{-2}]
    # Equation 21 [1]
    p.capGroPipe = 0.25*pi* p.lGroPipe*(( p.phiGroPipeE**2- p.phiGroPipeI**2)* p.rhoSteel*\
         p.cPSteel+ p.phiGroPipeI**2* p.rhoWater* p.cPWater)


    ## Lamp parameters - no lamps
    if noLamps:
        p.thetaLampMax = 0
        p.heatCorrection = 0
        p.etaLampPar = 0
        p.etaLampNir = 0
        p.tauLampPar = 1
        p.rhoLampPar = 0
        p.tauLampNir = 1
        p.rhoLampNir = 0
        p.tauLampFir = 1
        p.aLamp = 0
        p.epsLampTop = 0
        p.epsLampBottom = 0
        p.capLamp = 350
        p.cHecLampAir = 0
        p.etaLampCool = 0
        p.zetaLampPar = 0

    elif ledLamps:
        p.thetaLampMax = 200/3
        p.heatCorrection = 0
        p.etaLampPar = 3/5.41
        p.etaLampNir = 0.02
        p.tauLampPar = 0.98
        p.rhoLampPar = 0
        p.tauLampNir = 0.98
        p.rhoLampNir = 0
        p.tauLampFir = 0.98
        p.aLamp = 0.02
        p.epsLampTop = 0.88
        p.epsLampBottom = 0.88
        p.capLamp = 10
        p.cHecLampAir = 2.3
        p.etaLampCool = 0
        p.zetaLampPar = 5.41
        p.lampsOn = 0
        p.lampsOff = 18

    elif hpsLamps:
        p.thetaLampMax = 200/1.8    # Maximum intensity of lamps
        p.heatCorrection = 0        # correction for temperature setpoint when lamps are on
        p.etaLampPar = 1.8/4.9      # fraction of lamp input converted to PAR 			
        p.etaLampNir = 0.22         # fraction of lamp input converted to NIR 			
        p.tauLampPar = 0.98         # transmissivity of lamp layer to PAR 				
        p.rhoLampPar = 0            # reflectivity of lamp layer to PAR 				
        p.tauLampNir = 0.98         # transmissivity of lamp layer to NIR 				
        p.rhoLampNir = 0            # reflectivity of lamp layer to NIR 				
        p.tauLampFir = 0.98         # transmissivity of lamp layer to FIR 				
        p.aLamp = 0.02              # lamp area 										
        p.epsLampTop = 0.1          # emissivity of top side of lamp 					
        p.epsLampBottom = 0.9       # emissivity of bottom side of lamp 				
        p.capLamp = 100             # heat capacity of lamp 							
        p.cHecLampAir = 0.09        # heat exchange coefficient of lamp                       
        p.etaLampCool = 0           # fraction of lamp input removed by cooling               
        p.zetaLampPar = 4.9         # J to umol conversion of PAR output of lamp              
        p.lampsOn = 0               # Time of day when lamps go on                            
        p.lampsOff = 18             # Time of day when lamps go off                           

    # Interlight parameters - no lamps
    # currently we don't have interlights
    # however, since the controls do compute them, we need to set them to 0
    p.intLamps = intLamps
    p.vIntLampPos = 0.5
    p.fIntLampDown = 0.5
    p.capIntLamp = 10
    p.etaIntLampPar = 0
    p.etaIntLampNir = 0
    p.aIntLamp = 0
    p.epsIntLamp = 0
    p.thetaIntLampMax = 0
    p.zetaIntLampPar = 0
    p.cHecIntLampAir = 0
    p.tauIntLampFir = 1
    p.k1IntPar = 1.4
    p.k2IntPar = 1.4
    p.kIntNir = 0.54
    p.kIntFir = 1.88

    ## Other parameters
    p.cLeakTop = 0.5
    p.minWind = 0.25

    # p.dmfm = 0.0627
