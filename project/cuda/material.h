/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   noise.cu
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 27 Apr 2014
**
**************************************************************************/

#ifndef MATERIAL_CU
#define MATERIAL_CU

#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "math.h"
#include "common/math.h"
#include "cuda/vector.cu"

#define CUDA_INCLUDE

#include "cuda/noise.cu"
#include "sim/particle.h"
#include "sim/material.h"

__global__ void applyDefault(Particle *particles, int particleCount)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;
    particles[tid].material = MaterialConstants(); // default
}

__global__ void applyChunky(Particle *particles, int particleCount)
{
    // TODO - use the chunkiness parameter here to mix
    // spatially varying constitutive parameters
    // do this later...
    // snowballs are harder and heavier on the outside (stiffer, crunchier)
    // stiffness varied with noise fraction - chunky fracture.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;

    MaterialConstants mat = MaterialConstants();
    mat.lambda *= fbm3(particles[tid].position);
    particles[tid].material = mat;
    printf("%f\n",mat.lambda);

    //Particle particle = particles[tid];
    //fbm3(particle.position);
}

void applyMaterialPreset(Particle *particles, int particleCount, int preset)
{
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
    static const int threadCount = 128;

    if (preset == 1) // chunky
        applyChunky<<< numParticles/threadCount, threadCount >>>(particles,particleCount);
    else if (preset == 2) // hard/packed exterior, soft interior
        //applyChunky<<< numParticles/threadCount, threadCount >>>(particles,particleCount);
        int foo = 1; // do nothing for now
    else // default
        applyDefault<<< numParticles/threadCount, threadCount >>>(particles,particleCount);

    checkCudaErrors( cudaDeviceSynchronize() );
    printf("material preset %d applied \n", preset);
}

#endif

