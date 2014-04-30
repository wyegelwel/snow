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

#define THREAD_COUNT 128

#endif // HELPERS_H
