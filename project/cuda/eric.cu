/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   eric.cu
**   Author: evjang
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef ERIC_CU
#define ERIC_CU


#include <cuda.h>
#include <cuda_runtime.h> // prevents syntax errors on __global__ and __device__, among other things
#include <glm/geometric.hpp>
#include "math.h"   // this imports the CUDA math library

#define CUDA_INCLUDE    // only import the CUDA part of particle.h
#include "sim/particle.h"


#include <stdio.h>

extern "C"
void weightingTests();


__global__ void weightingTestKernel( );

void weightingTests()
{
    weightingTestKernel<<< 1,1 >>>();
}


__global__ void weightingTestKernel( )
{
    // compute weighting data on particles
    printf("Hello world!\n");
}


// non-testing stuff begins here


/**
 *  Computes SVD of
 */

__device__ void SVD( glm::mat3 *M, glm::mat3 *U, glm::mat3 *S, glm::mat3 *V )
{

}

/**
* Can get Polar Decomposition from SVD, see first section of http://en.wikipedia.org/wiki/Polar_decomposition,
*/
__device__ void polarD( glm::mat3 *M, glm::mat3 *R, glm::mat3 *S )
{

}



/**
 * 1D B-spline falloff
 * d is the distance from the point to the node center,
 * normalized by h such that particles <1 grid cell away
 * will have 0<d<1, particles >1 and <2 grid cells away will
 * still get some weight, and any particles further than that get
 * weight =0
 */
__device__ inline float N(float d)
{
    return (0<=d && d<1)*(.5*d*d*d-d*d+.666666) + (1<=d && d<2)*(-.166666f*d*d*d+d*d-2*d+1.333333f);
}


/**
 * sets w = interpolation weights (w_ip)
 * input is dx because we'd rather pre-compute abs outside so we can re-use again
 * in the weightGradient function.
 * by paper notation, w_ip = N_{i}^{h}(p) = N((xp-ih)/h)N((yp-jh)/h)N((zp-kh)/h)
 *
 *
 *
 */
__device__ void weight( glm::vec3 &dx, float h, float &w )
{
    w = N(dx.x/h)*N(dx.y/h)*N(dx.z/h);
}

/**
 * derivative of N with respect to d
 */
__device__ inline float Nd(float d)
{
    return (0 <= d && d<1)*(1.5f*d*d-2*d) + (1<=d && d<2)*(-.5*d*d+2*d-2);
}

/**
 * returns gradient of interpolation weights  \grad{w_ip}
 * xp = positions
 * dx = distance from grid cell
 * h =
 * wg =
 */
__device__ void weightGradient( glm::vec3 &xp, glm::vec3 &dx, float h, glm::vec3 &wg )
{
    float x = N(dx.x);float y = N(dx.y);float z = N(dx.z);
    glm::vec3 N = glm::vec3(x,y,z);
    glm::vec3 sgn = glm::sign(xp);
    glm::vec3 Nx = glm::vec3(Nd(dx.x)*sgn.x,Nd(dx.y)*sgn.y,Nd(dx.z)*sgn.z);

    wg.x = Nx.x * N.y * N.z;
    wg.y = N.x  * Nx.y* N.z;
    wg.z = N.x  * N.y * Nx.z;
}



#endif // ERIC_CU
