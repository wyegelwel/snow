/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   weighting.cu
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#ifndef WEIGHTING_H
#define WEIGHTING_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "glm/common.hpp"
#include "glm/vec3.hpp"

/*
 * 1D B-spline falloff
 * d is the distance from the point to the node center,
 * normalized by h such that particles <1 grid cell away
 * will have 0<d<1, particles >1 and <2 grid cells away will
 * still get some weight, and any particles further than that get
 * weight =0
 */
#define N( d )                                                                      \
({                                                                                  \
    __typeof__ (d) _d = (d);                                                        \
   ( ( 0 <= _d && _d < 1 ) * ( .5*_d*_d*_d - _d*_d + 2.f/3.f ) +                    \
     ( 1 <= _d && _d < 2 ) * ( -1.f/6.f*_d*_d*_d + _d*_d - 2*_d + 4.f/3.f ) );      \
})

/*
 * sets w = interpolation weights (w_ip)
 * input is dx because we'd rather pre-compute abs outside so we can re-use again
 * in the weightGradient function.
 * by paper notation, w_ip = N_{i}^{h}(p) = N((xp-ih)/h)N((yp-jh)/h)N((zp-kh)/h)
 */
__host__ __device__ __forceinline__ void weight( glm::vec3 &dx, float h, float &w )
{
    w = N( dx.x/h ) * N( dx.y/h ) * N( dx.z/h );
}

// Same as above, but dx is already normalized with h
__host__ __device__ __forceinline__ void weight( glm::vec3 &dx, float &w )
{
    w = N( dx.x ) * N( dx.y ) * N( dx.z );
}

/*
 * derivative of N with respect to d
 */
#define Nd( d )                                                                    \
({                                                                                 \
    __typeof__ (d) _d = (d);                                                       \
    ( ( 0 <= _d && _d < 1 ) * ( 1.5f*_d*_d - 2*_d ) +                              \
      ( 1 <= _d && _d < 2 ) * ( -.5*_d*_d + 2*_d - 2 ) );                          \
})

/*
 * returns gradient of interpolation weights  \grad{w_ip}
 * xp = sign( distance from grid node to particle )
 * dx = abs( distance from grid node to particle )
 */
__host__ __device__ __forceinline__ void weightGradient( const glm::vec3 &sdx, const glm::vec3 &dx, float h, glm::vec3 &wg )
{
    const glm::vec3 dx_h = dx / h;
    const glm::vec3 N = glm::vec3( N(dx_h.x), N(dx_h.y), N(dx_h.z) );
    const glm::vec3 Nx = sdx * glm::vec3( Nd(dx_h.x), Nd(dx_h.y), Nd(dx_h.z) );
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is already normalized with h
__host__ __device__ __forceinline__ void weightGradient( const glm::vec3 &sdx, const glm::vec3 &dx, glm::vec3 &wg )
{
    const glm::vec3 N = glm::vec3( N(dx.x), N(dx.y), N(dx.z) );
    const glm::vec3 Nx = sdx * glm::vec3( Nd(dx.x), Nd(dx.y), Nd(dx.z) );
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is not already absolute-valued
__host__ __device__ __forceinline__ void weightGradient( const glm::vec3 &dx, glm::vec3 &wg )
{
    const glm::vec3 sdx = glm::sign( dx );
    const glm::vec3 adx = glm::abs( dx );
    const glm::vec3 N = glm::vec3( N(adx.x), N(adx.y), N(adx.z) );
    const glm::vec3 Nx = sdx * glm::vec3( Nd(adx.x), Nd(adx.y), Nd(adx.z) );
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

/*
 * returns weight and gradient of weight, avoiding duplicate computations if applicable
 */
__host__ __device__ __forceinline__ void weightAndGradient( const glm::vec3 &sdx, const glm::vec3 &dx, float h, float &w, glm::vec3 &wg )
{
    const glm::vec3 dx_h = dx / h;
    const glm::vec3 N = glm::vec3( N(dx_h.x), N(dx_h.y), N(dx_h.z) );
    w = N.x * N.y * N.z;
    const glm::vec3 Nx = sdx * glm::vec3( Nd(dx_h.x), Nd(dx_h.y), Nd(dx_h.z) );
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is already normalized with h
__host__ __device__ __forceinline__ void weightAndGradient( const glm::vec3 &sdx, const glm::vec3 &dx, float &w, glm::vec3 &wg )
{
    const glm::vec3 N = glm::vec3( N(dx.x), N(dx.y), N(dx.z) );
    w = N.x * N.y * N.z;
    const glm::vec3 Nx = sdx * glm::vec3( Nd(dx.x), Nd(dx.y), Nd(dx.z) );
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is not already absolute-valued
__host__ __device__ __forceinline__ void weightAndGradient( const glm::vec3 &dx, float &w, glm::vec3 &wg )
{
    const glm::vec3 sdx = glm::sign( dx );
    const glm::vec3 adx = glm::abs( dx );
    const glm::vec3 N = glm::vec3( N(adx.x), N(adx.y), N(adx.z) );
    w = N.x * N.y * N.z;
    const glm::vec3 Nx = sdx * glm::vec3( Nd(adx.x), Nd(adx.y), Nd(adx.z) );
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

#endif // WEIGHTING_H
