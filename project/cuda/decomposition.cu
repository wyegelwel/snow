/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   decomposition.cu
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 13 Apr 2014
**
**************************************************************************/

#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "common/math.h"

#define GLM_FORCE_RADIANS
#include "glm/matrix.hpp"
#include "glm/mat3x3.hpp"
#include "glm/vec3.hpp"
#include "glm/gtc/matrix_access.hpp"
#include "glm/gtc/quaternion.hpp"

#define GAMMA 5.828427124 // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define CSTAR 0.923879532 // cos(pi/8)
#define SSTAR 0.3826834323 // sin(p/8)

extern "C"
{
    void svdTestsHost();
}


// Optimize C = glm::transpose(A) * B
__host__ __device__ inline void multTransposeL( const glm::mat3 &A, const glm::mat3 &B, glm::mat3 &C )
{
    glm::mat3 _C;
    _C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[0][1] + A[0][2]*B[0][2];
    _C[1][0] = A[0][0]*B[1][0] + A[0][1]*B[1][1] + A[0][2]*B[1][2];
    _C[2][0] = A[0][0]*B[2][0] + A[0][1]*B[2][1] + A[0][2]*B[2][2];
    _C[0][1] = A[1][0]*B[0][0] + A[1][1]*B[0][1] + A[1][2]*B[0][2];
    _C[1][1] = A[1][0]*B[1][0] + A[1][1]*B[1][1] + A[1][2]*B[1][2];
    _C[2][1] = A[1][0]*B[2][0] + A[1][1]*B[2][1] + A[1][2]*B[2][2];
    _C[0][2] = A[2][0]*B[0][0] + A[2][1]*B[0][1] + A[2][2]*B[0][2];
    _C[1][2] = A[2][0]*B[1][0] + A[2][1]*B[1][1] + A[2][2]*B[1][2];
    _C[2][2] = A[2][0]*B[2][0] + A[2][1]*B[2][1] + A[2][2]*B[2][2];
    C = _C;
}

__host__ __device__ void jacobiConjugation( int p, int q, glm::mat3 &S, glm::quat &qV )
{
    // eliminate off-diagonal entries Spq, Sqp
    float ch = 2.f * (S[0][0]-S[1][1]), ch2 = ch*ch;
    float sh = S[1][0], sh2 = sh*sh;
    bool flag = ( GAMMA * sh2 < ch2 );
    float w = rsqrt( ch2 + sh2 );
    ch = flag ? w*ch : CSTAR; ch2 = ch*ch;
    sh = flag ? w*sh : SSTAR; sh2 = sh*sh;

    // build rotation matrix Q
    glm::mat3 Q;
    float scale = ch*ch + sh*sh;
    float a = (ch2-sh2) / scale;
    float b = (2.f*sh*ch) / scale;
    Q[0][0] = a;  Q[1][0] = -b;
    Q[0][1] = b;  Q[1][1] = a;

    // perform the conjugation to annihilate S = Q' S Q
    multTransposeL( Q, S, S ); S *= Q;
    glm::vec3 tmp( qV.x, qV.y, qV.z );
    tmp *= sh;
    sh *= qV.w;
    // original
    qV *= ch;

    // terrible hack, this arranges such that for
    // (p,q) = ((0,1),(1,2),(0,2)), n = (0,1,2)
    int n = 2*q-p-2;
    int x = n;
    int y = (n+1)%3;
    int z = (n+2)%3;
    qV[z] += sh;
    qV.w -= tmp[z];
    qV[x] += tmp[y];
    qV[y] -= tmp[x];

    // re-arrange matrix for next iteration
    S = glm::mat3( S[1][1], S[1][2], S[1][0],
                   S[1][2], S[2][2], S[0][2],
                   S[1][0], S[0][2], S[0][0] );
}

/*
 * Wrapper function for the first step. Solve symmetric
 * eigenproblem using jacobi iteration. Given a symmetric
 * matrix S, diagonalize it also returns the cumulative
 * rotation as a quaternion.
 */
__host__ __device__ void jacobiEigenanalysis( glm::mat3 &S, glm::quat &qV )
{
    qV = glm::quat(1,0,0,0);
    for ( int sweep = 0; sweep < 4; ++sweep ) {
        // we wish to eliminate the maximum off-diagonal element
        // on every iteration, but cycling over all 3 possible rotations
        // in fixed order (p,q) = (1,2) , (2,3), (1,3) still has
        // asymptotic convergence
        jacobiConjugation( 0, 1, S, qV );
        jacobiConjugation( 1, 2, S, qV );
        jacobiConjugation( 0, 2, S, qV );
    }
}

// glm::toMat3 doesn't work in CUDA
__host__ __device__ void toMat3( const glm::quat &q, glm::mat3 &m )
{
    float qxx = q.x*q.x;
    float qyy = q.y*q.y;
    float qzz = q.z*q.z;
    float qxz = q.x*q.z;
    float qxy = q.x*q.y;
    float qyz = q.y*q.z;
    float qwx = q.w*q.x;
    float qwy = q.w*q.y;
    float qwz = q.w*q.z;
    m[0][0] = 1.f - 2.f*(qyy+qzz);
    m[0][1] = 2.f * (qxy+qwz);
    m[0][2] = 2.f * (qxz-qwy);
    m[1][0] = 2.f * (qxy-qwz);
    m[1][1] = 1.f - 2.f*(qxx+qzz);
    m[1][2] = 2.f * (qyz+qwx);
    m[2][0] = 2.f * (qxz+qwy);
    m[2][1] = 2.f * (qyz-qwx);
    m[2][2] = 1.f - 2.f*(qxx+qyy);
}

#define condSwap( COND, X, Y )          \
{                                       \
    __typeof__ (X) _X_ = X;             \
    bool _COND_ = (COND);               \
    X = _COND_ ? Y : X;                 \
    Y = _COND_ ? _X_ : Y;               \
}

#define condNegSwap( COND, X, Y )       \
{                                       \
    __typeof__ (X) _X_ = -X;            \
    bool _COND_ = (COND);               \
    X = _COND_ ? Y : X;                 \
    Y = _COND_ ? _X_ : Y;               \
}

__host__ __device__ void sortSingularValues( glm::mat3 &B, glm::mat3 &V )
{
    // used in step 2
    glm::vec3 b1 = glm::column(B,0); glm::vec3 v1 = glm::column(V,0);
    glm::vec3 b2 = glm::column(B,1); glm::vec3 v2 = glm::column(V,1);
    glm::vec3 b3 = glm::column(B,2); glm::vec3 v3 = glm::column(V,2);
    float rho1 = glm::dot(b1,b1);
    float rho2 = glm::dot(b2,b2);
    float rho3 = glm::dot(b3,b3);
    bool c;

    c = rho1 < rho2;
    condNegSwap( c, b1, b2 ); 
    condNegSwap( c, v1, v2 );
    condSwap( c, rho1, rho2 );

    c = rho1 < rho3;
    condNegSwap( c, b1, b3 ); 
    condNegSwap( c, v1, v3 );
    condSwap( c, rho1, rho3 );

    c = rho2 < rho3;
    condNegSwap( c, b2, b3 ); 
    condNegSwap( c, v2, v3 );

    // re-build B,V
    B = glm::mat3( b1, b2, b3 );
    V = glm::mat3( v1, v2, v3 );
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

__host__ __device__ void QRGivensQuaternion( float a1, float a2, float &ch, float &sh )
{
    /// TODO - if SVD isnt accurate enough, work on fixing accurateSQRT function here

    // a1 = pivot point on diagonal
    // a2 = lower triangular entry we want to annihilate

    // the authors be trippin, accurateSQRT doesn't work...
    //float rho = accurateSQRT(a1*a1 + a2*a2);
    float tmp = a1*a1 + a2*a2;
    float rho = tmp * rsqrt(tmp); // = sqrt(tmp)

    sh = rho > EPSILON ? a2 : 0;
    ch = fabsf(a1) + fmaxf( rho, EPSILON );
    bool b = a1 < 0;
    condSwap( b, sh, ch );
    //float w = glm::inversesqrt(ch*ch+sh*sh);
    //float w = glm::fastInverseSqrt(ch*ch+sh*sh);
    float w = rsqrt( ch*ch + sh*sh );

    ch *= w;
    sh *= w;
}

__host__ __device__ void QRDecomposition( const glm::mat3 &B, glm::mat3 &Q, glm::mat3 &R )
{
    R = B;

    // QR decomposition of 3x3 matrices using Givens rotations to
    // eliminate elements B21, B31, B32
    glm::quat qQ; // cumulative rotation
    glm::quat qU; // each Givens rotation in quaternion form
    glm::mat3 U;
    float ch, sh;

    // first givens rotation
    QRGivensQuaternion( R[0][0], R[0][1], ch, sh );
    qU = glm::quat( ch, 0, 0, sh );
    toMat3( qU, U );
    multTransposeL( U, R, R );

    // update cumulative rotation
    qQ *= qU;

    // second givens rotation
    QRGivensQuaternion( R[0][0], R[0][2], ch, sh );
    qU = glm::quat( ch, 0, -sh, 0 );
    toMat3( qU, U );
    multTransposeL( U, R, R );

    // update cumulative rotation
    qQ *= qU;

    // third Givens rotation
    QRGivensQuaternion( R[1][1], R[1][2], ch, sh );
    qU = glm::quat( ch, sh, 0, 0 );
    toMat3( qU, U );
    multTransposeL( U, R, R );

    // update cumulative rotation
    qQ *= qU;

    // qQ now contains final rotation for Q
    toMat3( qQ, Q );
}

/*
 * McAdams, Selle, Tamstorf, Teran, and Sifakis. Computing the Singular Value Decomposition of 3 x 3
 * matrices with minimal branching and elementary floating point operations
 * Computes SVD of 3x3 matrix A = W * S * V'
 */
__host__ __device__ void computeSVD( const glm::mat3 &A, glm::mat3 &W, glm::mat3 &S, glm::mat3 &V )
{
    // normal equations matrix
    glm::mat3 ATA;
    multTransposeL( A, A, ATA );

/// 2. Symmetric Eigenanlysis
    glm::quat qV;
    jacobiEigenanalysis( ATA, qV );
    toMat3( qV, V );
    glm::mat3 B = A * V; // B = AV = WS

/// 3. Sorting the singular values (find V)
    sortSingularValues( B, V );

/// 4. QR decomposition
    QRDecomposition( B, W, S );
}

/*
 * Returns polar decomposition of 3x3 matrix M where
 * M = Fe = Re * Se = U * P
 * U is an orthonormal matrix
 * S is symmetric positive semidefinite
 * Can get Polar Decomposition from SVD, see first section of http://en.wikipedia.org/wiki/Polar_decomposition
 */
__host__ __device__ void computePD( const glm::mat3 &A, glm::mat3 &U, glm::mat3 &P )
{
    // U is unitary matrix (i.e. orthogonal/orthonormal)
    // P is positive semidefinite Hermitian matrix
    glm::mat3 W, S, V;
    computeSVD( A, W, S, V );
    glm::mat3 Vt = glm::transpose(V);
    U = W * Vt;
    P = V * S * Vt;
}

/*
 * In snow we desire both SVD and polar decompositions simultaneously without
 * re-computing USV for polar.
 * here is a function that returns all the relevant values
 * SVD : A = W * S * V'
 * PD : A = U * P
 */
 __host__ __device__ void computeSVDandPD( const glm::mat3 &A, glm::mat3 &W, glm::mat3 &S, glm::mat3 &V, glm::mat3 &U, glm::mat3 &P )
 {
    computeSVD( A, W, S, V );
    glm::mat3 Vt = glm::transpose(V);
    U = W * Vt;
    P = V * S * Vt;
 }



 /**
  * TESTING ROUTINES
  */

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
     glm::mat3 A;
     // TEST 1
     A = glm::mat3(-0.558253  ,  -0.0461681   ,  -0.505735,
                   -0.411397 ,    0.0365854   ,   0.199707,
                   0.285389  ,   -0.313789   ,   0.200189);
     A = glm::transpose(A);

     svdTest<<<1,1>>>(A);
     cudaDeviceSynchronize();
}



#endif // DECOMPOSITION_H
