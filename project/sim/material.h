#ifndef MATERIAL_H
#define MATERIAL_H

#define POISSONS_RATIO 0.2

// YOUNGS MODULUS
#define E0 1.4e5 // default modulus
#define MIN_E0 4.8e4
#define MAX_E0 5e5

// CRITICAL COMPRESSION
#define MIN_THETA_C 1.9e-2
#define MAX_THETA_C 2.5e-2

// CRITICAL STRETCH
#define MIN_THETA_S 5e-3
#define MAX_THETA_S 7.5e-3

// HARDENING COEFF
#define MIN_XI 5
#define MAX_XI 20

#include <cuda.h>
#include <cuda_runtime.h>

struct Material
{
    float lambda; // first Lame parameter
    float mu; //second Lame paramter
    float xi; // Plastic hardening parameter

    // singular values restricted to [criticalCompression, criticalStretch]
    float criticalCompressionRatio;
    float criticalStretchRatio;

    // Constants from paper
    __host__ __device__ Material()
    {
        setYoungsAndPoissons( E0, POISSONS_RATIO );
        xi = 10;
        setCriticalStrains( MAX_THETA_C, MAX_THETA_S );
    }

    __host__ __device__
    Material( float compression,
              float stretch,
              float hardeningCoeff,
              float youngsModulus )
        : xi(hardeningCoeff),
          criticalCompressionRatio(compression),
          criticalStretchRatio(stretch)
    {
        setYoungsAndPoissons( youngsModulus, POISSONS_RATIO );
    }

    // Set constants in terms of Young's modulus and Poisson's ratio
    __host__ __device__
    void setYoungsAndPoissons( float E, float v )
    {
        lambda = (E*v)/((1+v)*(1-2*v));
        mu = E/(2*(1+v));
    }

    // Set constants in terms of Young's modulus and shear modulus (mu)
    __host__ __device__
    void setYoungsAndShear( float E, float G )
    {
        lambda = G*(E-2*G)/(3*G-E);
        mu = G;
    }

    // Set constants in terms of Lame's first parameter (lambda) and shear modulus (mu)
    __host__ __device__
    void setLameAndShear( float L, float G )
    {
        lambda = L;
        mu = G;
    }

    // Set constants in terms of Lame's first parameter (lambda) and Poisson's ratio
    __host__ __device__
    void setLameAndPoissons( float L, float v )
    {
        lambda = L;
        mu = L*(1-2*v)/(2*v);
    }

    // Set constants in terms of shear modulus (mu) and Poisson's ratio
    __host__ __device__
    void setShearAndPoissons( float G, float v )
    {
        lambda = (2*G*v)/(1-2*v);
        mu = G;
    }

    __host__ __device__
    void setCriticalCompressionStrain( float thetaC )
    {
        criticalCompressionRatio = 1.f - thetaC;
    }

    __host__ __device__
    void setCriticalStretchStrain( float thetaS )
    {
        criticalStretchRatio = 1.f + thetaS;
    }

    __host__ __device__
    void setCriticalStrains( float thetaC, float thetaS )
    {
        criticalCompressionRatio = 1.f - thetaC;
        criticalStretchRatio = 1.f + thetaS;
    }


};

#endif // MATERIAL_H
