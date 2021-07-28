# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
# distutils: language=c++

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "gtsne.h":
    cdef cppclass GTSNE:
        GTSNE()
        void run(double* X, double* Z, double* C, int N, int K, int D, int D_Z, double* Y, int no_dims, double alpha, double beta,
            double perplexity, double theta, unsigned int seed, int verbose)

cdef class ST_GTSNE:
    cdef GTSNE* thisptr # hold a C++ instance

    def __cinit__(self):
        self.thisptr = new GTSNE()

    def __dealloc__(self):
        del self.thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def run(self, X, Z, C, N, K, D, D_Z, d, alpha, beta, perplexity, theta, seed, verbose=False):
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _X = np.ascontiguousarray(X)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _Z = np.ascontiguousarray(Z)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _C = np.ascontiguousarray(C)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] Y = np.zeros((N, d), dtype=np.float64)
        self.thisptr.run(&_X[0,0], &_Z[0,0], &_C[0,0], N, K, D, D_Z, &Y[0,0], d, alpha, beta, perplexity, theta, seed, verbose)
        return Y
