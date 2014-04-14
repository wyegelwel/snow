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

#include "common/math.h"
#define GLM_FORCE_RADIANS
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"

#include "cuda/decomposition.cu"
#include "cuda/weighting.cu"

extern "C" {
void weightingTestsHost();
void svdTestsHost();
}

__device__ void printMat3(glm::mat3 mat) {
    // prints by rows
    for (int j=0; j<3; ++j) // g3d stores column-major
    {
        for (int i=0; i<3; ++i)
        {
            printf("%f   ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

__device__ void printQuat(glm::quat q)
{
    printf("%f  %f  %f  %f\n", q.w,q.x,q.y,q.z);
    //std::cout << q.w << "  " << q.x << "  " << q.y  << "  " << q.z <<std::endl;
    //std::cout << q[3] << "  " << q[0] << "  " << q[1]  << "  " << q[2] <<std::endl;
}

__device__ inline void printMat3Failure(char * name, glm::mat3 got, glm::mat3 expected)
{
    printf("%C GRADIENT: [FAILED] \n", name);
    printf("Expected: \n");
    printMat3(expected);
    printf("Got: \n");
    printMat3(got);
}

inline bool epsilonNotEqualMat3(glm::mat3 A, glm::mat3 B)
{
    for (int j=0;j<3;j++)
    {
        for (int i=0; i<3; i++)
        {
            if (fabs(A[i][j]-B[i][j]) >= EPSILON)
            {
                return true;
            }
        }
    }
    return false;
}

__global__ void svdTest(const glm::mat3 A)
{
    printf("original matrix\n");
    printMat3(A);
    glm::mat3 U1;
    glm::mat3 S;
    glm::mat3 V;
    glm::mat3 U2;
    glm::mat3 P;
    computeSVDandPD(A,U1,S,V,U2,P);

    glm::mat3 A_svd = U1*S*glm::transpose(V);
    glm::mat3 diff = glm::transpose(A_svd-A) * (A_svd-A);
    float norm = sqrtf( diff[0][0] + diff[1][1] + diff[2][2] );

    printf("SVD: U\n");
    printMat3(U1);
    printf("SVD: S\n");
    printMat3(S);
    printf("SVD: V\n");
    printMat3(V);
    printf("SVD: U*S*V'\n");
    printMat3(A_svd);
    printf("SVD: ||U*S*V' - A|| = %g\n\n", norm);
    printf("Polar: U\n");
    printMat3(U2);
    printf("Polar: P\n");
    printMat3(P);


}

void svdTestsHost()
{
    // multiple calls to svdTest
    glm::mat3 A;
    glm::mat3 U_e;
    glm::mat3 S_e;
    glm::mat3 V_e;
    // TEST 1
    A = glm::mat3(-0.558253  ,  -0.0461681   ,  -0.505735,
                  -0.411397 ,    0.0365854   ,   0.199707,
                  0.285389  ,   -0.313789   ,   0.200189);
    A = glm::transpose(A);

    svdTest<<<1,1>>>(A);
    cudaDeviceSynchronize();
}


/// EVERYTHING BELOW THIS IS RELATED TO THE WEIGHTING FUNCTIONS

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



#endif // ERIC_CU
