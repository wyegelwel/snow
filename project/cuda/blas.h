/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   blas.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 28 Apr 2014
**
**************************************************************************/

#ifndef BLAS_H
#define BLAS_H

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_INCLUDE
#include "cuda/vector.h"

__global__ void innerProductDotKernel( int length, const vec3 *u, const vec3 *v, float *dots )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= length ) return;
    dots[tid] = vec3::dot( u[tid], v[tid] );
}

__global__ void innerProductReduceKernel( int length, int reductionSize, float *dots )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= length || tid+reductionSize >= length ) return;
    dots[tid] += dots[tid+reductionSize];
}

__host__ float innerProduct( int length, const vec3 *u, const vec3 *v, float *dots )
{
    static const int threadCount = 100;
    static const dim3 blocks( (length+threadCount-1)/threadCount );
    static const dim3 threads( threadCount );

    innerProductDotKernel<<< blocks, threads >>>( length, u, v, dots );
    cudaDeviceSynchronize();

    int steps = (int)(ceilf(log2f(length)));
    int reductionSize = 1 << (steps-1);
    for ( int i = 0; i < steps; i++ ) {
        innerProductReduceKernel<<< blocks, threads >>>( length, reductionSize, dots );
        reductionSize /= 2;
        cudaDeviceSynchronize();
    }

    float result;
    cudaMemcpy( &result, dots, sizeof(float), cudaMemcpyDeviceToHost );
    return result;

}

__global__ void scaleAndAddKernel( int length, float sa, const vec3 *a, float sb, const vec3 *b, vec3 *c )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= length ) return;
    c[tid] = sa*a[tid] + sb*b[tid];
}

__host__ void scaleAndAdd( int length, float sa, const vec3 *a, float sb, const vec3 *b, vec3 *c )
{
    scaleAndAddKernel<<< (length+255)/256, 256 >>>( length, sa, a, sb, b, c );
    cudaDeviceSynchronize();
}

#endif // BLAS_H
