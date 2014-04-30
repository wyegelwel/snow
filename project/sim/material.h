#ifndef WORLD_H
#define WORLD_H

#define YOUNGS_MODULUS 1.4e5 // default modulus
#define POISSONS_RATIO 0.2

// YOUNGS MODULUS
#define MIN_E0 4.8e4
#define MAX_E0 1.4e5

// CRITICAL COMPRESSION
#define MIN_THETA_C 1.9e-2
#define MAX_THETA_C 2.5e-2

// CRITICAL STRETCH
#define MIN_THETA_S 5e-3
#define MAX_THETA_S 7.5e-3
// HARDENING COEFF
#define MIN_XI 5
#define MAX_XI 10


#include "cuda.h"
#include "cuda_runtime.h"


struct MaterialConstants
{
    float lambda; // first Lame parameter
    float mu; //second Lame paramter
    float xi; // Plastic hardening parameter
    float coeffFriction; // Coefficient of friction http://hypertextbook.com/facts/2007/TabraizRasul.shtml
    float criticalCompression; // singular values restricted to 1-theta_c, 1+theta_c
    float criticalStretch;

    __host__ __device__ MaterialConstants()
    {
        lambda = (YOUNGS_MODULUS*POISSONS_RATIO)/((1+POISSONS_RATIO)*(1-2*POISSONS_RATIO));
        mu = YOUNGS_MODULUS/(2*(1+POISSONS_RATIO));
        xi = 10;
        coeffFriction = 0.2;
        criticalCompression = 1.0 - MAX_THETA_C;
        criticalStretch = 1.0 + MAX_THETA_S;
    }

    __host__ __device__ MaterialConstants(float theta_c,
                                          float theta_s,
                                          float xi,
                                          float coeffFriction,
                                          float E0)
    : criticalCompression(1-theta_c),
      criticalStretch(1+theta_s),
      xi(xi)
    {
        // young's modulus is approximation for stiffness of material.
        lambda = (E0*POISSONS_RATIO)/((1-POISSONS_RATIO)*(1-2*POISSONS_RATIO));
        mu = E0/(2*(1+POISSONS_RATIO));
    }
};

#endif // WORLD_H
