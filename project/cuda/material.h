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
#include "cuda/vector.h"

#define CUDA_INCLUDE

#include "cuda/noise.h"
#include "sim/particle.h"
#include "sim/material.h"

// TODO - also copy MaterialConstants object from host to GPU so this can generate a distribution around that?
// Nah, this will be fixed presets.
// Presets override material settings unless we are on DEFAULT preset.
__global__ void applyChunky(Particle *particles, int particleCount)
{
    // TODO - use the chunkiness parameter here to mix
    // spatially varying constitutive parameters
    // snowballs are harder and heavier on the outside (stiffer, crunchier)
    // stiffness varied with noise fraction - chunky fracture.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;
    Particle &particle = particles[tid];
    vec3 pos = particle.position;

    float fbm = fbm3(pos);
    particle.material.lambda *= fbm;
//    printf("b : %f\n", lambda);

//    printf("a : %f\n", lambda);

    //Particle particle = particles[tid];
    //fbm3(particle.position);
}


#endif

