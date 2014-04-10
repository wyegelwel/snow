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

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h> // prevents syntax errors on __global__ and __device__, among other things
#include <glm/geometric.hpp>
#include "math.h"   // this imports the CUDA math library

#define CUDA_INCLUDE    // only import the CUDA part of particle.h
#include "sim/particle.h"


#define _EPSILON_ 1e-6
#define EPSILON _EPSILON_

#define EQ(a, b) (fabs((a) - (b)) < _EPSILON_)
#define NEQ(a, b) (fabs((a) - (b)) > _EPSILON_)



extern "C"
void weightingTestsHost();
void svdTestsHost();
void weightTest(float h, glm::vec3 ijk, glm::vec3 xp, float w_expected, glm::vec3 wg_expected);
void svdTest();

/**
 *  Computes SVD of
 */

__device__ void SVD( glm::mat3 *M, glm::mat3 *U, glm::mat3 *S, glm::mat3 *V )
{

}

/**weightingTestsHost
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
    return (0<=d && d<1)*(.5*d*d*d-d*d+2.f/3) + (1<=d && d<2)*(-1.f/6*d*d*d+d*d-2*d+4.f/3);
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

/// TESTING STUFF BEGINS HERE

__global__ void weightTestKernel(float h, glm::vec3 ijk, glm::vec3 xp, float w_expected, glm::vec3 wg_expected)
{
    // generate some points
    glm::vec3 g = ijk*h; // grid
    glm::vec3 dx = glm::abs(xp-g);

    float w;
    weight(dx,h,w);
    glm::vec3 wg;
    weightGradient(xp,dx,h,wg);

    // TODO - compare the values
    bool successw = true;
    bool successwg = true;
    //if (w != w_expected)
    bool fail = NEQ(w,w_expected);
    if (fail)
    {
        printf("WEIGHT : [FAILED] \n");
        printf("expected :%f\n", w_expected);
        printf("got : %f\n",w);
        successw = false;
    }
    fail = (NEQ(wg.x,wg_expected.x) || NEQ(wg.y,wg_expected.y) || NEQ(wg.z,wg_expected.z));
    if (fail)
    {
        printf("WEIGHT GRADIENT: [FAILED] \n");
        printf("expected : <%f,%f,%f>\n", wg_expected.x,wg_expected.y,wg_expected.z);
        printf("got : <%f,%f,%f>\n", wg.x,wg.y,wg.z);
        successwg = false;
    }
    if (successw && successwg)
    {
        printf("xp=<%f,%f,%f> : [PASSED]\n", xp.x,xp.y,xp.z);
    }
}


void weightTest(float h, glm::vec3 ijk, glm::vec3 xp, float w_expected, glm::vec3 wg_expected)
{
    weightTestKernel<<<1,1>>>(h,ijk,xp,w_expected,wg_expected);
    cudaDeviceSynchronize();
}

void weightingTestsHost()
{
    printf("beginning tests...\n");

    float h,w_expected;
    glm::vec3 ijk, xp,wg_expected;

    // TEST 1
    h = .1;
    ijk=glm::vec3(0,0,0);
    xp=glm::vec3(0,0,0);
    w_expected =  0.2962962;
    wg_expected=glm::vec3 (0,0,0);
    weightTest(h,ijk,xp,w_expected,wg_expected);

    // TEST 2
    h = .013;
    ijk=glm::vec3(12,10,3);
    xp=glm::vec3(4,4,.1);
    w_expected = 0;
    wg_expected=glm::vec3 (0,0,0);
    weightTest(h,ijk,xp,w_expected,wg_expected);

    // TEST 3
    h=1;
    ijk=glm::vec3(5,5,5);
    xp=glm::vec3(5,4.5,4.9);
    w_expected = 0.2099282;
    wg_expected=glm::vec3(0,-0.2738194,-0.05909722);
    weightTest(h,ijk,xp,w_expected,wg_expected);

    // TEST 4
    h=1;
    ijk=glm::vec3(1,2,3);
    xp=glm::vec3(1.1,2.2,3.3);
    w_expected = 0.2445964;
    wg_expected=glm::vec3 (-0.068856712,-0.13186487278,-0.192720697);
    weightTest(h,ijk,xp,w_expected,wg_expected);
}

void svdTestsHost()
{

}


#endif // ERIC_CU
