/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   max.cu
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef MAX_CU
#define MAX_CU

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define CUDA_INCLUDE
#include "tests/tests.h"
#include "cuda/vector.cu"

extern "C" { void implicitTests(); }

__global__ void innerProductDotKernel( int length, vec3 *u, vec3 *v, float *dots )
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

__host__ float innerProduct( int length, vec3 *u, vec3 *v, float *dots )
{
    static const int threadCount = 100;
    static const dim3 blocks( (length+threadCount-1)/threadCount );
    static const dim3 threads( threadCount );

    innerProductDotKernel<<< blocks, threads >>>( length, u, v, dots );
    checkCudaErrors( cudaDeviceSynchronize() );

    int steps = (int)(ceilf(log2f(length)));
    int reductionSize = 1 << (steps-1);
    for ( int i = 0; i < steps; i++ ) {
        innerProductReduceKernel<<< blocks, threads >>>( length, reductionSize, dots );
        reductionSize /= 2;
        checkCudaErrors( cudaDeviceSynchronize() );
    }

    float result;
    cudaMemcpy( &result, dots, sizeof(float), cudaMemcpyDeviceToHost );
    return result;

}

void testInnerProduct()
{

    printf( "INNER PRODUCT: " ); fflush(stdout);

    int N = 2000;
    vec3 *tmp = new vec3[2000];
    float expected = 0.f;
    for ( int i = 0; i < N; ++i ) {
        tmp[i] = vec3( i/2000.f, i/2000.f, i/2000.f );
        expected += vec3::dot( tmp[i], tmp[i] );
    }
    vec3 *u;
    cudaMalloc( (void**)&u, N*sizeof(vec3) );
    cudaMemcpy( u, tmp, N*sizeof(vec3), cudaMemcpyHostToDevice );
    delete [] tmp;

    float *dots;
    cudaMalloc( (void**)&dots, N*sizeof(float) );
    float result = innerProduct( N, u, u, dots );

    cudaFree( u );
    cudaFree( dots );

    if ( fabsf(result-expected) >= 1e-8 ) {
        printf( "FAILED - expected %g, got %g\n", expected, result );
        fflush( stdout );
    } else {
        printf( "PASSED\n" );
        fflush(stdout);
    }

}

void implicitTests()
{
    printf( "Starting Implicit tests...\n" ); fflush(stdout);

    testInnerProduct();

    printf( "Implicit tests done.\n" ); fflush(stdout);
}

#endif // MAX_CU
