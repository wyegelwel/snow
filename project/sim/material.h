#ifndef WORLD_H
#define WORLD_H

#define YOUNGS_MODULUS 1.4e5 // default modulus
#define POISSONS_RATIO 0.2

#include "cuda.h"
#include "cuda_runtime.h"


struct MaterialConstants
{
    float lambda; // first Lame parameter
    float mu; //second Lame paramter
    float xi; // Plastic hardening parameter
    float coeffFriction; // Coefficient of friction http://hypertextbook.com/facts/2007/TabraizRasul.shtml
    float criticalCompression;
    float criticalStretch;

    // Constants from paper

    __host__ __device__ MaterialConstants()
    {
        lambda = (YOUNGS_MODULUS*POISSONS_RATIO)/((1-POISSONS_RATIO)*(1-2*POISSONS_RATIO));
        mu = YOUNGS_MODULUS/(2*(1+POISSONS_RATIO));
        xi = 10;
        coeffFriction = 0.5; // XXX: FIND A GOOD ONE!
        criticalCompression = 1.0 - 2.5e-2;
        criticalStretch = 1.0 + 7.5e-3;
    }

    __host__ __device__ MaterialConstants(float critCompress,
                                          float critStretch,
                                          float hardeningCoeff,
                                          float coeffFriction,
                                          float youngsModulus)
    : criticalCompression(criticalCompression),
      criticalStretch(critStretch),
      xi(hardeningCoeff)
    {
        // young's modulus is approximation for stiffness of material.
        lambda = (youngsModulus*POISSONS_RATIO)/((1-POISSONS_RATIO)*(1-2*POISSONS_RATIO));
        mu = youngsModulus/(2*(1+POISSONS_RATIO));
    }
};

#endif // WORLD_H
