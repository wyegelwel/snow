/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   snowtypes.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 27 Apr 2014
**
**************************************************************************/

#ifndef SNOWTYPES_H
#define SNOWTYPES_H

#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "math.h"

#define CUDA_INCLUDE

#include "common/math.h"
#include "cuda/noise.h"
#include "cuda/vector.h"
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
    float fbm = fbm3( pos * 30.f ); // adjust the .5 to get desired frequency of chunks within fbm
    Material mat;
    mat.setYoungsAndPoissons( MIN_E0 + fbm*(MAX_E0-MIN_E0), POISSONS_RATIO );
    mat.xi = MIN_XI + fbm*(MAX_XI-MIN_XI);
    mat.setCriticalStrains( 5e-4, 1e-4 );
    particle.material = mat;
}

// hardening on the outside should be achieved with shells, so I guess this is the only spatially varying

#endif

