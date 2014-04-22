#ifndef WORLD_H
#define WORLD_H

#define YOUNGS_MODULUS 1.4e5
#define POISSONS_RATIO 0.2

#include "cuda.h"
#include "cuda_runtime.h"

struct MaterialConstants
{
    float lambda; // first Lame parameter
    float mu; //second Lame paramter
    float xi; // Plastic hardening parameter
    float coeffFriction; // Coefficient of friction
    float criticalCompression;
    float criticalStretch;

    // Constants from paper

    __host__ __device__ MaterialConstants()
    {
        lambda = (YOUNGS_MODULUS*POISSONS_RATIO)/((1-POISSONS_RATIO)*(1-2*POISSONS_RATIO));
        mu = YOUNGS_MODULUS/(2*(1+POISSONS_RATIO));
        xi = 10;
        coeffFriction = 1; // XXX: FIND A GOOD ONE!
        criticalCompression = 1.0 - 2.5e-2;
        criticalStretch = 1.0 + 7.5e-3;
    }

};

#endif // WORLD_H
