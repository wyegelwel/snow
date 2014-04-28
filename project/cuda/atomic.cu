/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   atomic.cu
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 28 Apr 2014
**
**************************************************************************/

#ifndef ATOMIC_H
#define ATOMIC_H

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_INCLUDE
#include "cuda/vector.cu"
#include "cuda/matrix.cu"

__device__ __forceinline__
void atomicAdd( vec3 *add, const vec3 &toAdd )
{
    atomicAdd(&(add->x), toAdd.x);
    atomicAdd(&(add->y), toAdd.y);
    atomicAdd(&(add->z), toAdd.z);
}

__device__ __forceinline__
void atomicAdd( mat3 *add, const mat3 &toAdd )
{
    atomicAdd(&(add->data[0]), toAdd[0]);
    atomicAdd(&(add->data[1]), toAdd[1]);
    atomicAdd(&(add->data[2]), toAdd[2]);
    atomicAdd(&(add->data[3]), toAdd[3]);
    atomicAdd(&(add->data[4]), toAdd[4]);
    atomicAdd(&(add->data[5]), toAdd[5]);
    atomicAdd(&(add->data[6]), toAdd[6]);
    atomicAdd(&(add->data[7]), toAdd[7]);
    atomicAdd(&(add->data[8]), toAdd[8]);
}

#endif // ATOMIC_H
