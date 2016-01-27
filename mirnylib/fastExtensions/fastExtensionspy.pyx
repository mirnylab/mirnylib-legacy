import numpy as np
cimport numpy as np
import cython
cimport cython
from cpython cimport bool

ctypedef fused real:
    cython.char
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.float
    cython.double



cdef extern from "fastExtensions.h":
    cdef T openmmSum[T](T* a, T* b)



cdef extern from "fastExtensions.h":
    void  readWigFileCpp(char* filename, double* data, int chromCount,
                            bool useX, bool useY, bool useM,
                            int Xnum, int Ynum, int Mnum, int Mkb, int resolution)





def openmmArraySum(real[:] a):
    return openmmSum(&a[0], &a[a.shape[0]-1])



def readWigFile(char* filename, double [:] data, int chromCount,
                            bool useX, bool useY, bool useM,
                            int Xnum, int Ynum, int Mnum, int Mkb, int resolution):

    readWigFileCpp(filename, &data[0], chromCount,
                             useX,  useY,  useM,
                             Xnum,  Ynum,  Mnum,  Mkb,  resolution)
    pass



