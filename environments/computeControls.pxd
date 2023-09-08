from auxiliaryStates cimport AuxiliaryStates
from defineParameters cimport Parameters
from utils cimport satVp, co2dens2ppm
from libc.math cimport exp, log, fmax, fmin, floor

cdef inline double proportionalControl(processVar, setPt, pBand, minVal, maxVal):
    return minVal + (maxVal - minVal)*(1/(1+exp(-2/pBand*log(100)*(processVar - setPt - pBand/2))))

cdef inline double* controlSignal(Parameters* p, double* x, double* u, double* d):
    cdef double timeOfDay
    cdef double dayOfYear
    cdef double lampTimeOfDay
    cdef double lampDayOfYear
    cdef double lampNoCons
    cdef double linearLampSwitchOn
    cdef double linearLampSwitchOff
    cdef double linearLampBothSwitches
    cdef double smoothLamp
    cdef double isDayInside
    cdef double heatSetPoint
    cdef double heatMax
    cdef double co2SetPoint
    cdef double co2InPpm
    cdef double ventHeat
    cdef double rhIn
    cdef double ventRh
    cdef double ventCold
    cdef double thScrSp
    cdef double thScrCold
    cdef double thScrHeat
    cdef double thScrRh

    timeOfDay = 24*(x[27] - floor(x[27])) # hours since midnight time of day [h]
    dayOfYear = x[27] % 365.2425          


    # Control of the lamp according to the time of day [0/1]
    # if p.lampsOn < p.lampsOff, lamps are on from p.lampsOn to p.lampsOff each day
    # if p.lampsOn > p.lampsOff, lamps are on from p.lampsOn until p.lampsOff the next day
    # if p.lampsOn == p.lampsOff, lamps are always off
    # for continuous light, set p.lampsOn = -1, p.lampsOff = 25
    # addAux(gl, 'lampTimeOfDay', ((p.lampsOn<=p.lampsOff).* ...
    #     (p.lampsOn < gl.timeOfDay & gl.timeOfDay < p.lampsOff) + ... 
        # (1-(p.lampsOn<=p.lampsOff)).*(p.lampsOn<gl.timeOfDay | gl.timeOfDay<p.lampsOff))...
    #     .*1); % multiply by 1 to convert from logical to double
    lampTimeOfDay = ((p.lampsOn <= p.lampsOff) * (p.lampsOn < timeOfDay and timeOfDay < p.lampsOff) + \
                        (1-(p.lampsOn <= p.lampsOff)) * (p.lampsOn < timeOfDay or timeOfDay < p.lampsOff))

    # CURRENTLY UNUSED...
    # Control of the lamp according to the day of year [0/1]
    # if p.dayLampStart < p.dayLampStop, lamps are on from p.dayLampStart to p.dayLampStop
    # if p.dayLampStart > p.dayLampStop, lamps are on from p.lampsOn until p.lampsOff the next year
    # if p.dayLampStart == p.dayLampStop, lamps are always off
    # for no influence of day of year, set p.dayLampStart = -1, p.dayLampStop > 366
    lampDayOfYear = ((p.dayLampStart <= p.dayLampStop) * (p.dayLampStart < dayOfYear and dayOfYear < p.dayLampStop) + \
                        (1-(p.dayLampStart <= p.dayLampStop)) * (p.dayLampStart < dayOfYear or dayOfYear < p.dayLampStop))


    # THIS VARIABLE MAINLY REPRESENTS WHETHER WE ARE IN LIGHT PERIOD OF THE GREENHOUSE
    # Control for the lamps disregarding temperature and humidity constraints
    # Chapter 4 Section 2.3.2, Chapter 5 Section 2.4 [7]
    # Section 2.3.2 [8]
    # This variable is used to decide if the greenhouse is in the light period
    # ("day inside"), needed to set the climate setpoints. 
    # However, the lamps may be switched off if it is too hot or too humid
    # in the greenhouse. In this case, the greenhouse is still considered
    # to be in the light period
    # addAux(gl, 'lampNoCons', 1.*(gl.d.iGlob < gl.p.lampsOffSun).* ... # lamps are off if sun is not too bright
    #     (gl.d.dayRadSum < gl.p.lampRadSumLimit).* ... # and the predicted daily radiation sum is less than the predefined limit 
    #     gl.lampTimeOfDay.* ... # and the time of day is within the lighting period
    #     gl.lampDayOfYear); # and the day of year is within the lighting season
    lampNoCons = (d[0] < p.lampsOffSun) * (d[7] < p.lampRadSumLimit) * lampTimeOfDay * lampDayOfYear

    ## Smoothing of control of the lamps
    # To allow smooth transition between day and night setpoints

    # Linear version of lamp switching on: 
    # 1 at lampOn, 0 one hour before lampOn, with linear transition
    # Note: this current function doesn't do a linear interpolation if
    # lampOn == 0
    linearLampSwitchOn = fmax(0, fmin(1, timeOfDay-p.lampsOn + 1))

    # Linear version of lamp switching on: 
    # 1 at lampOff, 0 one hour after lampOff, with linear transition
    # Note: this current function doesn't do a linear interpolation if
    # lampOff == 24
    linearLampSwitchOff = fmax(0, fmin(1, p.lampsOff - timeOfDay + 1))

    # Combination of linear transitions above
    # if p.lampsOn < p.lampsOff, take the minimum of the above
    # if p.lampsOn > p.lampsOn, take the maximum
    # if p.lampsOn == p.lampsOff, set at 0
    # addAux(gl, 'linearLampBothSwitches', ...
    #     (p.lampsOn~=p.lampsOff).*((p.lampsOn<p.lampsOff).*min(gl.linearLampSwitchOn,gl.linearLampSwitchOff) ...
    #     + (1-(p.lampsOn<p.lampsOff)).*max(gl.linearLampSwitchOn,gl.linearLampSwitchOff)));
    linearLampBothSwitches = (p.lampsOn!=p.lampsOff)*((p.lampsOn<p.lampsOff)*min(linearLampSwitchOn,linearLampSwitchOff)
        + (1-(p.lampsOn<p.lampsOff))*fmax(linearLampSwitchOn,linearLampSwitchOff))

    # Smooth (linear) approximation of the lamp control
    # To allow smooth transition between light period and dark period setpoints
    # 1 when lamps are on, 0 when lamps are off, with a linear
    # interpolation in between
    # Does not take into account the lamp switching off due to 
    # instantaenous sun radiation, excess heat or humidity
    # addAux(gl, 'smoothLamp', gl.linearLampBothSwitches.* ... # linear transition between lamp on and off
    #     (gl.d.dayRadSum < gl.p.lampRadSumLimit).* ... # lamps off if the predicted daily radiation sum is more than the predefined limit 
    #     gl.lampDayOfYear); # lamps off if day of year is not within the lighting season
    smoothLamp = linearLampBothSwitches * (d[7] < p.lampRadSumLimit) * lampDayOfYear

    # Indicates whether daytime climate settings should be used, i.e., if
    # the sun is out or the lamps are on
    # 1 if day, 0 if night. If lamps are on it is considered day
    isDayInside = fmax(smoothLamp, d[8])

    # Heating set point [°C]
    heatSetPoint = isDayInside*p.tSpDay + (1-isDayInside)*p.tSpNight + p.heatCorrection*lampNoCons

    #% Ventilation setpoint due to excess heating set point [°C]
    heatMax = heatSetPoint + p.heatDeadZone

    # CO2 set point [ppm]
    co2SetPoint = isDayInside*p.co2SpDay

    # CO2 concentration in main compartment [ppm]
    co2InPpm = co2dens2ppm(x[2], 1e-6*x[0])

    # Ventilation setpoint due to excess heating set point [°C]
    ventHeat = proportionalControl(x[2], heatMax, p.ventHeatPband, 0, 1)

    # Relative humidity [%]
    rhIn = 100*x[15]/satVp(x[2])

    # Ventilation setpoint due to excess humidity [°C]
    # mechallowed = 1 if mechanical ventilation is allowed, 0 otherwise We have have it at zero
    ventRh = proportionalControl(rhIn, p.rhMax+0 * p.mechDehumidPband, p.ventRhPband, 0, 1)

    # Ventilation closure due to too cold temperatures 
    ventCold = proportionalControl(x[2], heatSetPoint-p.tVentOff, p.ventColdPband, 1, 0)

    # Setpoint for closing the thermal screen [°C]
    thScrSp = (d[8]) * p.thScrSpDay + (1- (d[8])) * p.thScrSpNight

    # Closure of the thermal screen based on outdoor temperature [0-1, 0 is fully open]
    thScrCold = proportionalControl(d[1], thScrSp, p.thScrPband, 0, 1)

    # Opening of thermal screen closure due to too high temperatures 
    thScrHeat = proportionalControl(x[2], heatSetPoint+p.thScrDeadZone, -p.thScrPband, 1, 0)

    # Opening of thermal screen due to high humidity [0-1, 0 is fully open]
    thScrRh = fmax(proportionalControl(rhIn, p.rhMax+p.thScrRh, p.thScrRhPband, 1, 0), 1-ventCold)
        # if 1-ventCold == 0 (it's too cold inside to ventilate)
        # don't force to open the screen (even if RH says it should be 0)
        # Better to reduce RH by increasing temperature

    # Control for the top lights: 
    # 1 if lamps are on, 0 if lamps are off
    # addAux(gl, 'lampOn', gl.lampNoCons.* ... # Lamps should be on
    #     proportionalControl(gl.x.tAir, gl.heatMax+gl.p.lampExtraHeat, -0.5, 0, 1).* ... # Switch lamp off if too hot inside
    #     ...                                            # Humidity: only switch off if blackout screen is used 
    #     (gl.d.isDaySmooth + (1-gl.d.isDaySmooth).* ... # Blackout sceen is only used at night 
    #         max(proportionalControl(gl.rhIn, gl.p.rhMax+gl.p.blScrExtraRh, -0.5, 0, 1),... # Switch lamp off if too humid inside
    #                     1-gl.ventCold))); # Unless ventCold == 0
                        # if ventCold is 0 it's too cold inside to ventilate, 
                        # better to raise the RH by heating. 
                        # So don't open the blackout screen and 
                        # don't stop illuminating in this case. 
    lampOn = lampNoCons * proportionalControl(x[2], heatMax + p.lampExtraHeat, -0.5, 0, 1) *\
                (d[9] + (1-d[9])) *\
                fmax(proportionalControl(rhIn, p.rhMax + p.blScrExtraRh, -0.5, 0, 1), 1-ventCold)

    # Control for the interlights: 
    # 1 if interlights are on, 0 if interlights are off
    # addAux(gl, 'intLampOn', gl.lampNoCons.* ... # Lamps should be on
    #     proportionalControl(gl.x.tAir, gl.heatMax+gl.p.lampExtraHeat, -0.5, 0, 1).* ... # Switch lamp off if too hot inside
    #     ... # Humidity: only switch off if blackout screen is used 
    #     (gl.d.isDaySmooth + (1-gl.d.isDaySmooth).* ... # Blackout screen is only used at night 
    #         max(proportionalControl(gl.rhIn, gl.p.rhMax+gl.p.blScrExtraRh, -0.5, 0, 1),... # Switch lamp off if too humid inside
    #                     1-gl.ventCold))); 
                        # if ventCold is 0 it's too cold inside to ventilate, 
                        # better to raise the RH by heating. 
                        # So don't open the blackout screen and 
                        # don't stop illuminating in this case. 
    intLampOn = lampNoCons * proportionalControl(x[2], heatMax + p.lampExtraHeat, -0.5, 0, 1) * \
                (d[9] + (1-d[9])) *\
                fmax(proportionalControl(rhIn, p.rhMax + p.blScrExtraRh, -0.5, 0, 1), 1-ventCold)


    # boiler, co2, thscr, roof, lamps, intlamps, boilgro, blscr
    u[0] = proportionalControl(x[2], heatSetPoint, p.tHeatBand, 0, 1)
    u[1] = proportionalControl(co2InPpm, co2SetPoint, p.co2Band, 0, 1)
    u[2] = fmin(thScrCold, fmax(thScrHeat, thScrRh))
    u[3] = fmin(ventCold, fmax(ventHeat, ventRh))
    u[4] = lampOn
    u[5] = intLampOn
    u[6] = proportionalControl(x[2], heatSetPoint, p.tHeatBand, 0, 1)
    u[7] = p.useBlScr * (1-d[9]) * fmax(lampOn, intLampOn)

    # UNUSED shading screen, permanent shading screen, side ventilation
    # u[8] = 0
    # u[9] = 0
    # u[10] = 0
    return u