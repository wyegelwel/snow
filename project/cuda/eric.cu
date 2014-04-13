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
#define GLM_FORCE_RADIANS
#include "glm/geometric.hpp"
#include "glm/gtc/epsilon.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
//#include "glm/gtx/fast_square_root.hpp" // doesnt work with CUDA
#include "glm/gtc/matrix_access.hpp"

#include "math.h"




#define _sqrtHalf 0.70710678
#define _gamma 5.828427124 // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define _cstar 0.923879532 // cos(pi/8)
#define _sstar 0.3826834323 // sin(p/8)


#define _EPSILON_ 1e-6
#define EPSILON _EPSILON_

#define EQ(a, b) (fabs((a) - (b)) < _EPSILON_)
#define NEQ(a, b) (fabs((a) - (b)) > _EPSILON_)


extern "C"
{
void weightingTestsHost();
void svdTestsHost();
}
//void weightTest(float h, glm::vec3 ijk, glm::vec3 xp, float w_expected, glm::vec3 wg_expected);
//void svdTest();


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

__device__ void toMat3(const glm::quat q, glm::mat3 &m)
{
    // glm::toMat3 doesn't work in CUDA
    float qxx = q.x*q.x;
    float qyy = q.y*q.y;
    float qzz = q.z*q.z;
    float qxz = q.x*q.z;
    float qxy = q.x * q.y;
    float qyz=q.y * q.z;
    float qwx=q.w * q.x;
    float qwy=q.w * q.y;
    float qwz=q.w * q.z;
    m[0][0] = 1 - 2 * (qyy +  qzz);
    m[0][1] = 2 * (qxy + qwz);
    m[0][2] = 2 * (qxz - qwy);
    m[1][0] = 2 * (qxy - qwz);
    m[1][1] = 1 - 2 * (qxx +  qzz);
    m[1][2] = 2 * (qyz + qwx);
    m[2][0] = 2 * (qxz + qwy);
    m[2][1] = 2 * (qyz - qwx);
    m[2][2] = 1 - 2 * (qxx +  qyy);
}


//inline float accurateRSQRT(float x)
//{
//    // TODO - something is wrong with this.
//    // used in step 3
//    /* Lomont 2003 */
//    //float y = glm::fastSqrt(x);
//    //return y * (3-x*y*y)/2;
//}

//inline float accurateSQRT(float x)
//{
//    return x * accurateRSQRT(x);
//}

__device__ void condSwap(bool c, float &X, float &Y)
{
    // used in step 2
    float Z = X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}

// swapping functions for entire rows
__device__ void condSwap(const bool c, glm::vec3 &X, glm::vec3 &Y)
{
    // used in step 2
    glm::vec3 Z = X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}

__device__ void condNegSwap(const bool c, glm::vec3 &X, glm::vec3 &Y)
{
    // used in step 2 and 3
    glm::vec3 Z = -X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}

// __device__ void condNegSwap(const bool c, int c1, int c2, glm::quat &qV)
// {
//     // condNegSwap can be modified to operate well
//     // on quaternion representation of V
//     glm::quat qR;
//     // qR = (1, 0, 0, c) for 1,2
//     // what are the other quaternions corresponding to the other rot matrices?
//     qV = qV * qR;
// }

__device__ void approximateGivensQuaternion(float a11, float a12, float a22, float &ch, float &sh)
{
    /*
     * Given givens angle computed by approximateGivensAngles,
     * compute the corresponding rotation quaternion.
     * returns ch, sh. Up to the user to build the corresponding quaternion.
     */
    ch = 2*(a11-a22);
    sh = a12;
    bool b = _gamma*sh*sh < ch*ch;
    //float w = glm::fastInverseSqrt(ch*ch+sh*sh);
    float w = rsqrt(ch*ch+sh*sh);

    ch=b?w*ch:_cstar;
    sh=b?w*sh:_sstar;
    
}

__device__ void jacobiConjugation(const int p, const int q, glm::mat3 &S, glm::quat &qV)
{
    // eliminate off-diagonal entries Spq, Sqp
    float ch,sh;
    approximateGivensQuaternion(S[0][0],S[1][0],S[1][1],ch,sh);
    // build rotation matrix Q
    glm::mat3 Q;
    float scale = ch*ch+sh*sh;
    float a = (ch*ch-sh*sh)/scale;
    float b = (2*sh*ch)/scale;
    Q[0][0] = a;    Q[1][0] = -b;
    Q[0][1] = b;    Q[1][1] = a;
    // perform the conjugation to annihilate S = Q' S Q
    S =glm::transpose(Q) * S * Q;
    glm::vec3 tmp(qV.x,qV.y,qV.z);
    tmp *= sh;
    sh *= qV.w;
    // original
    qV *= ch;
    int n = 2*q-p-2; // terrible hack 
    // this arranges such that for (p,q) = ((0,1),(1,2),(0,2)),
    // n = (0,1,2)    
    int x = n;
    int y = (n+1)%3;
    int z = (n+2)%3;
    qV[z] += sh;
    qV.w -= tmp[z];
    qV[x] += tmp[y];
    qV[y] -= tmp[x];
    // re-arrange matrix for next iteration
    S = glm::mat3(S[1][1], S[1][2], S[1][0],
                  S[1][2], S[2][2], S[0][2],
                  S[1][0], S[0][2], S[0][0]);
}

__device__ void sortSingularValues(glm::mat3 &B, glm::mat3 &V)
{
    // used in step 2
    glm::vec3 b1 = glm::column(B,0); glm::vec3 v1 = glm::column(V,0);
    glm::vec3 b2 = glm::column(B,1); glm::vec3 v2 = glm::column(V,1);
    glm::vec3 b3 = glm::column(B,2); glm::vec3 v3 = glm::column(V,2);
    float rho1 = glm::length2(b1);
    float rho2 = glm::length2(b2);
    float rho3 = glm::length2(b3);
    bool c;

    c = rho1 < rho2;
    condNegSwap(c,b1,b2); condNegSwap(c,v1,v2);
    condSwap(c,rho1,rho2);

    c = rho1 < rho3;
    condNegSwap(c,b1,b3); condNegSwap(c,v1,v3);
    condSwap(c,rho1,rho3);

    c = rho2 < rho3;
    condNegSwap(c,b2,b3); condNegSwap(c,v2,v3);

    // re-build B,V
    B = glm::mat3(b1,b2,b3);
    V = glm::mat3(v1,v2,v3);
}


__device__ void QRGivensQuaternion(float a1, float a2, float &ch, float &sh)
{
    /// TODO - if SVD isnt accurate enough, work on fixing accurateSQRT function here

    // a1 = pivot point on diagonal
    // a2 = lower triangular entry we want to annihilate
    float epsilon = EPSILON;
    // the authors be trippin, accurateSQRT doesn't work...
    //float rho = accurateSQRT(a1*a1 + a2*a2);
    float tmp = a1*a1+a2*a2;
    float rho = tmp*rsqrt(tmp); // = sqrt(tmp)
    //float rho = rsqrt(a1*a1 + a2*a2);

    sh = rho > epsilon ? a2 : 0;
    ch = fabs(a1) + fmax(rho,epsilon);
    bool b = a1 < 0;
    condSwap(b,sh,ch);
    //float w = glm::inversesqrt(ch*ch+sh*sh);
    //float w = glm::fastInverseSqrt(ch*ch+sh*sh);
    float w = rsqrt(ch*ch+sh*sh);

    ch *= w;
    sh *= w;
}

__device__ void QRDecomposition(glm::mat3 B, glm::mat3 &Q, glm::mat3 &R)
{
    // QR decomposition of 3x3 matrices using Givens rotations to
    // eliminate elements B21, B31, B32
    glm::quat qQ; // cumulative rotation
    glm::quat qU; // each Givens rotation in quaternion form
    glm::mat3 U;
    float ch, sh;
    QRGivensQuaternion(B[0][0],B[0][1],ch,sh);
    qU = glm::quat(ch,0,0,sh);
    //U = glm::toMat3(qU);
    toMat3(qU,U);
    B = glm::transpose(U) * B;

    // update cumulative rotation
    qQ *= qU;

    // second givens rotation
    QRGivensQuaternion(B[0][0],B[0][2],ch,sh);
    qU = glm::quat(ch,0,-sh,0);
    //U = glm::toMat3(qU);
    toMat3(qU,U);
    B = glm::transpose(U) * B;
    qQ *= qU;

    // third Givens rotation
    QRGivensQuaternion(B[1][1],B[1][2],ch,sh);
    qU = glm::quat(ch,sh,0,0);
    //U = glm::toMat3(qU);
    toMat3(qU,U);
    B = glm::transpose(U) * B;
    qQ *= qU;
    // B has been transformed into R
    R = B;
    // qQ now contains final rotation for Q
    //Q = glm::toMat3(qQ);
    toMat3(qQ,Q);
}

__device__ void jacobiEigenanalysis(glm::mat3 &S, glm::quat &qV)
{
    // wrapper function for the first step
    // solve symmetric eigenproblem using jacobi iteration
    // given a symmetric matrix S, diagonalize it
    // also returns the cumulative rotation as a quaternion
    qV = glm::quat(1,0,0,0);
    for(int sweep=0;sweep<4;sweep++)
    {
        // we wish to eliminate the maximum off-diagonal element
        // on every iteration, but cycling over all 3 possible rotations
        // in fixed order (p,q) = (1,2) , (2,3), (1,3) still has
        //  asymptotic convergence
        jacobiConjugation(0,1,S,qV);
        jacobiConjugation(1,2,S,qV);
        jacobiConjugation(0,2,S,qV);
    }
}

/*
 * McAdams, Selle, Tamstorf, Teran, and Sifakis. Computing the Singular Value Decomposition of 3 x 3
 * matrices with minimal branching and elementary floating point operations
 * Computes SVD of 3x3 matrix A = U * S * V'
 */
__device__ void SVD(const glm::mat3 A, glm::mat3 &U, glm::mat3 &S, glm::mat3 &V )
{
    // normal equations matrix
    glm::mat3 ATA = glm::transpose(A) * A;
/// 2. Symmetric Eigenanlysis
    glm::quat qV;
    jacobiEigenanalysis(ATA,qV);
    //V = glm::toMat3(qV);
    toMat3(qV,V);
    glm::mat3 B = A * V; // B = AV = US
/// 3. Sorting the singular values (find V)
    sortSingularValues(B,V);
/// 4. QR decomposition
    QRDecomposition(B,U,S);
}

/**
 * Returns polar decomposition of 3x3 matrix M where
 * M = Fe = Re * Se = U * P
 * R is the orthonormal matrix, S is the rotation
 * S is symmetric positive semidefinite
 * Can get Polar Decomposition from SVD, see first section of http://en.wikipedia.org/wiki/Polar_decomposition
 */
__device__ void polarD(const glm::mat3 A, glm::mat3 &U, glm::mat3 &P )
{
    // U is unitary matrix (i.e. orthogonal/orthonormal)
    // P is positive semidefinite Hermitian matrix
    glm::mat3 U1, S, V;
    SVD(A,U1,S,V);
    glm::mat3 Vt = glm::transpose(V);
    U = U1 * Vt; 
    P = V * S * Vt; 
}

/*
 * In snow we desire both SVD and polar decompositions simultaneously without
 * re-computing USV for polar.
 * here is a function that returns all the relevant values
 * SVD : A = U1 * S * V'
 * PD : A = U2 * P
 */
 __device__ void svdAndPolarD(const glm::mat3 A, glm::mat3 &U1, glm::mat3 &S, glm::mat3 &V, glm::mat3 &U2, glm::mat3 &P)
 {
    SVD(A,U1,S,V);
    glm::mat3 Vt = glm::transpose(V);
    U2 = U1 * Vt; 
    P = V * S * Vt; 
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
    svdAndPolarD(A,U1,S,V,U2,P);

    printf("SVD: U\n");
    printMat3(U1);
    printf("SVD: S\n");
    printMat3(S);
    printf("SVD: V\n");
    printMat3(V);
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




/// EVERYTHING BELOW THIS IS RELAT TO THE WEIGHTING FUNCTIONS


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
