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


/*
 * theta_c, theta_s -> determine when snow starts breaking.
 *          larger = chunky, wet. smaller = powdery, dry
 *
 * low xi, E0 = muddy. high xi, E0 = Icy
 * low xi = ductile, high xi = brittle
 *
 */

__global__ void applyChunky(Particle *particles, int particleCount)
{
    // spatially varying constitutive parameters
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;
    Particle &particle = particles[tid];
    vec3 pos = particle.position;
    float fbm = fbm3(pos*.5); // adjust the .5 to get desired frequency of chunks within fbm
    MaterialConstants mat(MAX_THETA_C,MAX_THETA_S,MIN_XI+fbm*(MAX_XI-MIN_XI),0.2,MIN_E0+fbm*(MAX_E0-MIN_E0));
    particle.material = mat;
}

// hardening on the outside should be achieved with shells, so I guess this is the only spatially varying

#endif

