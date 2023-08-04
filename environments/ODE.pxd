from defineParameters cimport Parameters
from auxiliaryStates cimport AuxiliaryStates, update
from libc.stdlib cimport malloc, free


cdef inline double* ODE(AuxiliaryStates* a, Parameters* p, double* x, double* u, double* d, char nx):
    # cdef double *x_new = (double*)malloc(27 * sizeof(double))
    cdef double* ki = <double*>malloc(nx * sizeof(double))

    # update auxiliary states

    # Carbon concentration of greenhouse air [mg m^{-3} s^{-1}]
    # setOde(gl, 'co2Air', 1/p.capCo2Air*(a.mcBlowAir+a.mcExtAir+a.mcPadAir ...
    #     -a.mcAirCan-a.mcAirTop-a.mcAirOut));
    ki[0] = 1/p.capCo2Air * (a.mcBlowAir+a.mcExtAir+a.mcPadAir-a.mcAirCan-a.mcAirTop-a.mcAirOut)    
    
    # Carbon concentration of top compartment [mg m^{-3} s^{-1}]
    # setOde(gl, 'co2Top', 1/p.capCo2Top*(a.mcAirTop-a.mcTopOut));
    ki[1] = 1/p.capCo2Top * (a.mcAirTop-a.mcTopOut)

    # % Greenhouse air temperature [�C s^{-1}]
    # setOde(gl, 'tAir', 1/p.capAir*(a.hCanAir+a.hPadAir-a.hAirMech+a.hPipeAir ...
    #     +a.hPasAir+a.hBlowAir+a.rGlobSunAir-a.hAirFlr-a.hAirThScr-a.hAirOut ...
    #     -a.hAirTop-a.hAirOutPad-a.lAirFog - a.hAirBlScr ...
    #     +a.hLampAir+a.rLampAir ...
    #     +a.hGroPipeAir+a.hIntLampAir+a.rIntLampAir));
    ki[2] = 1/p.capAir * (a.hCanAir+a.hPadAir-a.hAirMech+a.hPipeAir \
        +a.hPasAir+a.hBlowAir+a.rGlobSunAir-a.hAirFlr-a.hAirThScr-a.hAirOut \
        -a.hAirTop-a.hAirOutPad-a.lAirFog-a.hAirBlScr \
        +a.hLampAir+a.rLampAir \
        +a.hGroPipeAir+a.hIntLampAir+a.rIntLampAir)

    # # % Air above screen temperature [�C s^{-1}]
    # # setOde(gl, 'tTop', 1/p.capTop*(a.hThScrTop+a.hAirTop-a.hTopCovIn-a.hTopOut+a.hBlScrTop));
    ki[3] = 1/p.capTop * (a.hThScrTop+a.hAirTop-a.hTopCovIn-a.hTopOut+a.hBlScrTop)

    # # % Canopy temperature [�C s^{-1}]
    # # setOde(gl, 'tCan', ...
    # # (1./a.capCan).*(a.rParSunCan+a.rNirSunCan+a.rPipeCan ...
    # #     -a.hCanAir-a.lCanAir-a.rCanCovIn-a.rCanFlr-a.rCanSky-a.rCanThScr-a.rCanBlScr ...
    # #     +a.rParLampCan+a.rNirLampCan+a.rFirLampCan ...
    # #     +a.rGroPipeCan ... 
    # #     +a.rParIntLampCan+a.rNirIntLampCan+a.rFirIntLampCan)); 
    ki[4] = 1/a.capCan * (a.rParSunCan+a.rNirSunCan+a.rPipeCan \
        -a.hCanAir-a.lCanAir-a.rCanCovIn-a.rCanFlr-a.rCanSky-a.rCanThScr-a.rCanBlScr \
        +a.rParLampCan+a.rNirLampCan+a.rFirLampCan \
        +a.rGroPipeCan+a.rParIntLampCan+a.rNirIntLampCan+a.rFirIntLampCan)

    # # % Internal cover temperature [�C s^{-1}]
    # #   setOde(gl, 'tCovIn', (1./a.capCovIn).*(a.hTopCovIn+a.lTopCovIn+a.rCanCovIn+ ...
    # #       a.rFlrCovIn+a.rPipeCovIn+a.rThScrCovIn-a.hCovInCovE+ ...
    #     #   a.rLampCovIn+a.rBlScrCovIn+a.rIntLampCovIn));
    ki[5] = (1/a.capCovIn) * (a.hTopCovIn+a.lTopCovIn+a.rCanCovIn+ \
        a.rFlrCovIn+a.rPipeCovIn+a.rThScrCovIn-a.hCovInCovE+ \
        a.rLampCovIn+a.rBlScrCovIn+a.rIntLampCovIn)

    # % External cover temperature [�C s^{-1}]
    # % Equation 8 [1]
    #   setOde(gl, 'tCovE', (1./a.capCovE).*(a.rGlobSunCovE+a.hCovInCovE-a.hCovEOut-a.rCovESky));
    ki[6] = (1/a.capCovE) * (a.rGlobSunCovE+a.hCovInCovE-a.hCovEOut-a.rCovESky)

    # % Thermal screen temperature [�C s^{-1}]
    # % Equation 5 [1], Equation A1 [5], Equation 7.1 [6]
    # setOde(gl, 'tThScr', (1/p.capThScr)*(a.hAirThScr+a.lAirThScr+a.rCanThScr+ ...
    #     a.rFlrThScr+a.rPipeThScr-a.hThScrTop-a.rThScrCovIn-a.rThScrSky+a.rBlScrThScr+ ...
    #     a.rLampThScr+a.rIntLampThScr));
    ki[7] = (1/p.capThScr)*(a.hAirThScr+a.lAirThScr+a.rCanThScr+ \
        a.rFlrThScr+a.rPipeThScr-a.hThScrTop-a.rThScrCovIn-a.rThScrSky+a.rBlScrThScr+ \
        a.rLampThScr+a.rIntLampThScr)

    # % Greenhouse floor temperature [�C s^{-1}]
    # % Equation 3 [1], Equation A1 [5], Equation 7.1 [6]
    # setOde(gl, 'tFlr', 1/p.capFlr*(a.hAirFlr+a.rParSunFlr+a.rNirSunFlr ...
    #     +a.rCanFlr+a.rPipeFlr-a.hFlrSo1-a.rFlrCovIn-a.rFlrSky-a.rFlrThScr...
    #     +a.rParLampFlr+a.rNirLampFlr+a.rFirLampFlr-a.rFlrBlScr...
    #     +a.rParIntLampFlr+a.rNirIntLampFlr+a.rFirIntLampFlr));
    ki[8] = 1/p.capFlr*(a.hAirFlr+a.rParSunFlr+a.rNirSunFlr \
        +a.rCanFlr+a.rPipeFlr-a.hFlrSo1-a.rFlrCovIn-a.rFlrSky-a.rFlrThScr \
        +a.rParLampFlr+a.rNirLampFlr+a.rFirLampFlr-a.rFlrBlScr \
        +a.rParIntLampFlr+a.rNirIntLampFlr+a.rFirIntLampFlr)

    # Pipe temperature [�C s^{-1}]
    # % Equation 9 [1], Equation A1 [5], Equation 7.1 [6]
    # setOde(gl, 'tPipe', 1/p.capPipe*(a.hBoilPipe+a.hIndPipe+a.hGeoPipe-a.rPipeSky ...
    # -a.rPipeCovIn-a.rPipeCan-a.rPipeFlr-a.rPipeThScr-a.hPipeAir ...
    # +a.rLampPipe-a.rPipeBlScr+a.hBufHotPipe+a.rIntLampPipe));
    ki[9] = 1/p.capPipe*(a.hBoilPipe+a.hIndPipe+a.hGeoPipe-a.rPipeSky \
    -a.rPipeCovIn-a.rPipeCan-a.rPipeFlr-a.rPipeThScr-a.hPipeAir \
    +a.rLampPipe-a.rPipeBlScr+a.hBufHotPipe+a.rIntLampPipe)

    # Soil layer 1 temperature [�C s^{-1}]
    # setOde(gl, 'tSo1', 1/p.capSo1*(a.hFlrSo1-a.hSo1So2));
    ki[10] = 1/p.capSo1*(a.hFlrSo1-a.hSo1So2)

    # Soil layer 2 temperature [�C s^{-1}]
    # setOde(gl, 'tSo2', 1/p.capSo2*(a.hSo1So2-a.hSo2So3));
    ki[11] = 1/p.capSo2*(a.hSo1So2-a.hSo2So3)

    # Soil layer 3 temperature [�C s^{-1}]
    # setOde(gl, 'tSo3', 1/p.capSo3*(a.hSo2So3-a.hSo3So4));
    ki[12] = 1/p.capSo3*(a.hSo2So3-a.hSo3So4)

    # Soil layer 4 temperature [�C s^{-1}]
    # setOde(gl, 'tSo4', 1/p.capSo4*(a.hSo3So4-a.hSo4So5));
    ki[13] = 1/p.capSo4*(a.hSo3So4-a.hSo4So5)

    # Soil layer 5 temperature [�C s^{-1}]
    # setOde(gl, 'tSo5', 1/p.capSo5*(a.hSo4So5-a.hSo5SoOut));
    ki[14] = 1/p.capSo5*(a.hSo4So5-a.hSo5SoOut)

    ## Vapor balance
    
    # Vapor pressure of greenhouse air [Pa s^{-1}] = [kg m^{-1} s^{-3}]
    # Equation 10 [1], Equation A40 [5], Equation 7.40, 7.50 [6]
    # setOde(gl, 'vpAir', (1./a.capVpAir).*(a.mvCanAir+a.mvPadAir+a.mvFogAir+a.mvBlowAir ...
    #     -a.mvAirThScr-a.mvAirTop-a.mvAirOut-a.mvAirOutPad-a.mvAirMech-a.mvAirBlScr));
    ki[15] = (1/a.capVpAir) * (a.mvCanAir+a.mvPadAir+a.mvFogAir+a.mvBlowAir \
        -a.mvAirThScr-a.mvAirTop-a.mvAirOut-a.mvAirOutPad-a.mvAirMech-a.mvAirBlScr)

    # Vapor pressure of air in top compartment [Pa s^{-1}] = [kg m^{-1} s^{-3}]
    # Equation 11 [1]
    # setOde(gl, 'vpTop', (1./a.capVpTop).*(a.mvAirTop-a.mvTopCovIn-a.mvTopOut));
    ki[16] = (1/a.capVpTop) * (a.mvAirTop-a.mvTopCovIn-a.mvTopOut)

    # Lamp temperature [�C s^{-1}]
    # Equation A1 [5], Equation 7.1 [6]
    # setOde(gl, 'tLamp', 1/p.capLamp*(a.qLampIn-a.hLampAir-a.rLampSky-a.rLampCovIn ...
    #     -a.rLampThScr-a.rLampPipe-a.rLampAir - a.rLampBlScr ...
    #     -a.rParLampFlr-a.rNirLampFlr-a.rFirLampFlr ...
    #     -a.rParLampCan-a.rNirLampCan-a.rFirLampCan-a.hLampCool+a.rIntLampLamp));
    ki[17] = 1/p.capLamp * (a.qLampIn-a.hLampAir-a.rLampSky-a.rLampCovIn \
        -a.rLampThScr-a.rLampPipe-a.rLampAir - a.rLampBlScr \
        -a.rParLampFlr-a.rNirLampFlr-a.rFirLampFlr \
        -a.rParLampCan-a.rNirLampCan-a.rFirLampCan-a.hLampCool+a.rIntLampLamp)

    # Interlight temperature [�C s^{-1}]
    # Equation 7.1 [6]
    # setOde(gl, 'tIntLamp', 1/p.capIntLamp*(a.qIntLampIn-a.hIntLampAir-a.rIntLampSky-a.rIntLampCovIn ...
    #     -a.rIntLampThScr-a.rIntLampPipe-a.rIntLampAir-a.rIntLampBlScr ...
    #     -a.rParIntLampFlr-a.rNirIntLampFlr-a.rFirIntLampFlr ...
    #     -a.rParIntLampCan-a.rNirIntLampCan-a.rFirIntLampCan-a.rIntLampLamp));
    ki[18] = 1/p.capIntLamp * (a.qIntLampIn-a.hIntLampAir-a.rIntLampSky-a.rIntLampCovIn \
        -a.rIntLampThScr-a.rIntLampPipe-a.rIntLampAir-a.rIntLampBlScr \
        -a.rParIntLampFlr-a.rNirIntLampFlr-a.rFirIntLampFlr \
        -a.rParIntLampCan-a.rNirIntLampCan-a.rFirIntLampCan-a.rIntLampLamp)

    # Grow pipes temperature [�C s^{-1}]
    # setOde(gl, 'tGroPipe', 1/p.capGroPipe*(a.hBoilGroPipe-a.rGroPipeCan-a.hGroPipeAir));
    ki[19] = 1/p.capGroPipe * (a.hBoilGroPipe-a.rGroPipeCan-a.hGroPipeAir)

    # % Blackout screen temperature [�C s^{-1}]
    # % Equation A1 [5], Equation 7.1 [6]
    # setOde(gl, 'tBlScr', (1/p.capBlScr)*(a.hAirBlScr+a.lAirBlScr+a.rCanBlScr+ ...
    #     a.rFlrBlScr+a.rPipeBlScr-a.hBlScrTop-a.rBlScrCovIn-a.rBlScrSky-a.rBlScrThScr+ ...
    #     a.rLampBlScr+a.rIntLampBlScr));
    ki[20] = (1/p.capBlScr) * (a.hAirBlScr+a.lAirBlScr+a.rCanBlScr+ \
        a.rFlrBlScr+a.rPipeBlScr-a.hBlScrTop-a.rBlScrCovIn-a.rBlScrSky-a.rBlScrThScr+ \
        a.rLampBlScr+a.rIntLampBlScr)

    # % Average canopy temperature in last 24 hours
    # % Equation 9 [2]
    # setOde(gl, 'tCan24', 1/86400*(x.tCan-x.tCan24));
    ki[21] = 1/86400*(x[4]-x[21])

    ## Crop model [2]
    
    # Carbohydrates in buffer [mg{CH2O} m^{-2} s^{-1}]
    # Equation 1 [2]
    # setOde(gl, 'cBuf', a.mcAirBuf-a.mcBufFruit-a.mcBufLeaf-a.mcBufStem-a.mcBufAir);
    ki[22] = a.mcAirBuf-a.mcBufFruit-a.mcBufLeaf-a.mcBufStem-a.mcBufAir

    # Carbohydrates in leaves [mg{CH2O} m^{-2} s^{-1}]
    # Equation 4 [2]
    # setOde(gl, 'cLeaf', a.mcBufLeaf - a.mcLeafAir - a.mcLeafHar);
    ki[23] = a.mcBufLeaf-a.mcLeafAir-a.mcLeafHar

    # Carbohydrates in stem [mg{CH2O} m^{-2} s^{-1}]
    # Equation 6 [2]
    # setOde(gl, 'cStem', a.mcBufStem - a.mcStemAir);
    ki[24] = a.mcBufStem-a.mcStemAir

    # Carbohydrates in fruit [mg{CH2O} m^{-2} s^{-1}]
    # Equation 2 [2], Equation A44 [5]
    # setOde(gl, 'cFruit', a.mcBufFruit - a.mcFruitAir - a.mcFruitHar);
    ki[25] = a.mcBufFruit-a.mcFruitAir-a.mcFruitHar

    # Crop development stage [�C day s^{-1}]
    # Equation 8 [2]
    # setOde(gl, 'tCanSum', 1/86400*x.tCan);
    ki[26] = 1/86400*x[4]

    return ki
