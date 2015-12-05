#include <omp.h>

#ifndef __fastExtension_H__
#define __fastExtension_H__
void readWigFileCpp(char* filename, double* data, int chromCount,
                            bool useX, bool useY, bool useM,
                            int Xnum, int Ynum, int Mnum, int Mkb, int resolution);

template <typename TYPE>
TYPE openmmSum(TYPE* a, TYPE* b)
{
    TYPE sum;
    TYPE* i ; 
    #pragma omp parallel for      \
    default(shared) private(i)  \
    reduction(+:sum)
    for (i=a; i<=b; i++)
          sum += *i;
    return sum;
};



                          
#endif
