# from define_parameters cimport Parameters
from define_parameters_old cimport Parameters
# from auxiliary_states cimport AuxiliaryStates, update
from auxiliary_states_old cimport AuxiliaryStates, update


from compute_controls cimport controlSignal
from ODE cimport ODE
from libc.stdlib cimport malloc, free
from libc.math cimport isnan

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
