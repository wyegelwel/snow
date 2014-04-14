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
#define N( d ) ( ( 0 <= d && d < 1 ) * ( .5*d*d*d - d*d + 2.f/3.f ) + \
                 ( 1 <= d && d < 2 ) * ( -1.f/6.f*d*d*d + d*d - 2*d + 4.f/3.f ) )


/*
 * sets w = interpolation weights (w_ip)
 * input is dx because we'd rather pre-compute abs outside so we can re-use again
 * in the weightGradient function.
 * by paper notation, w_ip = N_{i}^{h}(p) = N((xp-ih)/h)N((yp-jh)/h)N((zp-kh)/h)
 */
__device__ void weight( glm::vec3 &dx, float h, float &w )
{
    w = N( dx.x/h ) * N( dx.y/h ) * N( dx.z/h );
}

/*
 * derivative of N with respect to d
 */
#define Nd( d ) ( ( 0 <= d && d < 1 ) * ( 1.5f*d*d - 2*d ) + \
                  ( 1 <= d && d < 2 ) * ( -.5*d*d + 2*d - 2 ) )

/*
 * returns gradient of interpolation weights  \grad{w_ip}
 * xp = positions
 * dx = distance from grid cell
 * h =
 * wg =
 */
__device__ void weightGradient( const glm::vec3 &xp, const glm::vec3 &dx, float h, glm::vec3 &wg )
{
    glm::vec3 N = glm::vec3( N(dx.x), N(dx.y), N(dx.z) );
    glm::vec3 Nx = glm::sign(xp) * glm::vec3( Nd(dx.x), Nd(dx.y), Nd(dx.z) );
    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}

#endif // WEIGHTING_H
