from define_parameters cimport Parameters
from auxiliary_states cimport AuxiliaryStates, update


from compute_controls cimport controlSignal
from ODE cimport ODE
from libc.stdlib cimport malloc, free
from libc.math cimport isnan, sqrt, fabs

cdef inline double* fRK4(AuxiliaryStates* a, Parameters* p, double* u, double* x, double* d, float h, char nx):
    """
    Difference function that computes the next state.
    """
    cdef double* k1
    cdef double* k2
    cdef double* k3
    cdef double* k4

    cdef double* x2 = <double*>malloc(nx * sizeof(double))
    cdef double* x3 = <double*>malloc(nx * sizeof(double))
    cdef double* x4 = <double*>malloc(nx * sizeof(double))

    cdef unsigned char i
    cdef unsigned char j
    cdef unsigned char k
    cdef unsigned char l

    # update auxiliary states
    update(a, p, u, x, d)

    # comptures the harvested fruit over the timestep
    a.mcFruitHarSum += a.mcFruitHar*h

    k1 = ODE(a, p, x, u, d, nx)

    for i in range(nx):
        x2[i] = x[i] + h/2*k1[i]

    update(a, p, u, x2, d)
    k2 = ODE(a, p, x2, u, d, nx)

    for j in range(nx):
        x3[j] = x[j] + h/2*k2[j]

    update(a, p, u, x3, d)
    k3 = ODE(a, p, x3, u, d, nx)

    for k in range(nx):
        x4[k] = x[k] + h*k3[k]

    update(a, p, u, x4, d)
    k4 = ODE(a, p, x4, u, d, nx)

    # Runge-Kutta 4th order method
    for l in range(nx):
        x[l] += h/6 * (k1[l] + 2*k2[l] + 2*k3[l] + k4[l])

    free(k1)
    free(k2)
    free(k3)
    free(k4)

    free(x2)
    free(x3)
    free(x4)
    return x

cdef inline double* fRK45(AuxiliaryStates* a, Parameters* p, double* u, double* x, double* d, double h, char nx, unsigned int timestep):
    """
    Difference function that computes the next state.
    """
    cdef double* k1
    cdef double* k2
    cdef double* k3
    cdef double* k4
    cdef double* k5
    cdef double* k6

    # cdef double* xnew 
    cdef double* x2 = <double*>malloc(nx * sizeof(double))
    cdef double* x3 = <double*>malloc(nx * sizeof(double))
    cdef double* x4 = <double*>malloc(nx * sizeof(double))
    cdef double* x5 = <double*>malloc(nx * sizeof(double))
    cdef double* x6 = <double*>malloc(nx * sizeof(double))

    cdef unsigned char i
    cdef unsigned char j
    cdef unsigned char k
    cdef unsigned char l
    cdef unsigned char m
    cdef unsigned char n
    cdef unsigned char o
    cdef unsigned char q

    cdef double b11=1/5
    cdef double b21=3/40
    cdef double b31=44/45
    cdef double b41=19372/6561
    cdef double b51=9017/3168
    cdef double b61=35/384
    cdef double b22=9/40
    cdef double b32=-56/15
    cdef double b42=-25360/2187
    cdef double b52=-355/33
    cdef double b33=32/9
    cdef double b43=64448/6561
    cdef double b53=46732/5247
    cdef double b63=500/1113
    cdef double b44=-212/729
    cdef double b54=49/176
    cdef double b64=125/192
    cdef double b55=-5103/18656
    cdef double b65=-2187/6784
    cdef double b66=11/84

    # update auxiliary states
    update(a, p, u, x, d)

    # variable to keep track of the harvested fruit per timestep
    # Actually, this is the euler method...
    a.mcFruitHarSum += a.mcFruitHar*h

    k1 = ODE(a, p, x, u, d, nx)

    for i in range(nx):
        x2[i] = x[i] + h*(b11*k1[i])

    update(a, p, u, x2, d)
    k2 = ODE(a, p, x2, u, d, nx)

    for j in range(nx):
        x3[j] = x[j] + h*(b21*k1[j] + b22*k2[j])

    update(a, p, u, x3, d)
    k3 = ODE(a, p, x3, u, d, nx)

    for k in range(nx):
        x4[k] = x[k] + h*(b31*k1[k] + b32*k2[k] + b33*k3[k])

    update(a, p, u, x4, d)
    k4 = ODE(a, p, x4, u, d, nx)

    for l in range(nx):
        x5[l] = x[l] + h*(b41*k1[l] + b42*k2[l] + b43*k3[l] + b44*k4[l])

    update(a, p, u, x5, d)
    k5 = ODE(a, p, x5, u, d, nx)

    for m in range(nx):
        x6[m] = x[m] + h*(b51*k1[m] + b52*k2[m] + b53*k3[m] + b54*k4[m] + b55*k5[m])

    update(a, p, u, x6, d)
    k6 = ODE(a, p, x6, u, d, nx)

    # Runge-Kutta 6th order method
    for n in range(nx):
        x[n] = x[n] + h*(b61*k1[n] + b63*k3[n] + b64*k4[n] + b65*k5[n] + b66*k6[n])

    free(k1)
    free(k2)
    free(k3)
    free(k4)
    free(k5)
    free(k6)

    free(x2)
    free(x3)
    free(x4)
    free(x5)
    free(x6)
    return x


cdef inline double* fEuler(AuxiliaryStates* a, Parameters* p, double* u, double* x, double* d, float h, char nx):
    """
    Difference function that computes the next state.
    """
    cdef double* k1
    cdef unsigned char l

    # update auxiliary states
    update(a, p, u, x, d)
    k1 = ODE(a, p, x, u, d, nx)

    # Forward Euler
    for l in range(nx):
        x[l] += h * (k1[l])

    free(k1)
    return x
