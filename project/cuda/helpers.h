/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   helpers.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 30 Apr 2014
**
**************************************************************************/

#ifndef HELPERS_H
#define HELPERS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define LAUNCH( ... ) { __VA_ARGS__; checkCudaErrors( cudaDeviceSynchronize() ); }

#define cudaMallocAndCopy( dst, src, size )                    \
({                                                             \
    cudaMalloc((void**) &dst, size);                           \
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);        \
})

#define TEST( toCheck, msg, failExprs)      \
({                                          \
    if (toCheck){                           \
        printf("[PASSED]: %s\n", msg);      \
    }else{                                  \
        printf("[FAILED]: %s\n", msg);      \
        failExprs;                          \
    }                                       \
})

#define THREAD_COUNT 128

#endif // HELPERS_H
