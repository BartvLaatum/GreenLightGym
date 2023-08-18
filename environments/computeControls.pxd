from libc.math cimport exp log

cdef inline double* proprtionalControl(processVar, setPt, pBand, minVal, maxVal):
    return minVal + (maxVal - minVal)*(1/(1+exp(-2/pBand*log(100)*(processVar - setPt - pBand/2))))

