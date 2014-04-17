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

#include "math.h"
#include "common/math.h"

#include "cuda/matrix.cu"

#define GAMMA 5.828427124 // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define CSTAR 0.923879532 // cos(pi/8)
#define SSTAR 0.3826834323 // sin(p/8)

__host__ __device__ void jacobiConjugation( int x, int y, int z, mat3 &S, quat &qV )
{
    // eliminate off-diagonal entries Spq, Sqp
    float ch = 2.f * (S[0]-S[4]), ch2 = ch*ch;
    float sh = S[3], sh2 = sh*sh;
    bool flag = ( GAMMA * sh2 < ch2 );
    float w = rsqrtf( ch2 + sh2 );
    ch = flag ? w*ch : CSTAR; ch2 = ch*ch;
    sh = flag ? w*sh : SSTAR; sh2 = sh*sh;

    // build rotation matrix Q
    float scale = 1.f / (ch2 + sh2);
    float a = (ch2-sh2) * scale;
    float b = (2.f*sh*ch) * scale;
    float a2 = a*a, b2 = b*b, ab = a*b;

    // Use what we know about Q to simplify S = Q' * S * Q
    // and the re-arranging step.
    float s0 = a2*S[0] + 2*ab*S[1] + b2*S[4];
    float s2 = a*S[2] + b*S[5];
    float s3 = (a2-b2)*S[1] + ab*(S[4]-S[0]);
    float s4 = b2*S[0] - 2*ab*S[1] + a2*S[4];
    float s5 = a*S[7] - b*S[6];
    float s8 = S[8];
    S = mat3( s4, s5, s3,
              s5, s8, s2,
              s3, s2, s0 );

    vec3 tmp( sh*qV.x, sh*qV.y, sh*qV.z );
    sh *= qV.w;
    // original
    qV *= ch;

    qV[z] += sh;
    qV.w -= tmp[z];
    qV[x] += tmp[y];
    qV[y] -= tmp[x];
}

/*
 * Wrapper function for the first step. Solve symmetric
 * eigenproblem using jacobi iteration. Given a symmetric
 * matrix S, diagonalize it also returns the cumulative
 * rotation as a quaternion.
 */
__host__ __device__ __forceinline__ void jacobiEigenanalysis( mat3 &S, quat &qV )
{
    qV = glm::quat( 1,0,0,0 );

    jacobiConjugation( 0, 1, 2, S, qV );
    jacobiConjugation( 1, 2, 0, S, qV );
    jacobiConjugation( 2, 0, 1, S, qV );

    jacobiConjugation( 0, 1, 2, S, qV );
    jacobiConjugation( 1, 2, 0, S, qV );
    jacobiConjugation( 2, 0, 1, S, qV );

    jacobiConjugation( 0, 1, 2, S, qV );
    jacobiConjugation( 1, 2, 0, S, qV );
    jacobiConjugation( 2, 0, 1, S, qV );

    jacobiConjugation( 0, 1, 2, S, qV );
    jacobiConjugation( 1, 2, 0, S, qV );
    jacobiConjugation( 2, 0, 1, S, qV );
}

#define condSwap( COND, X, Y )          \
{                                       \
    __typeof__ (X) _X_ = X;             \
    X = COND ? Y : X;                   \
    Y = COND ? _X_ : Y;                 \
}

#define condNegSwap( COND, X, Y )       \
{                                       \
    __typeof__ (X) _X_ = -X;            \
    X = COND ? Y : X;                   \
    Y = COND ? _X_ : Y;                 \
}

__host__ __device__ __forceinline__ void sortSingularValues( mat3 &B, mat3 &V )
{
    // used in step 2
    vec3 b1 = B.column(0); vec3 v1 = V.column(0);
    vec3 b2 = B.column(1); vec3 v2 = V.column(1);
    vec3 b3 = B.column(2); vec3 v3 = V.column(2);
    float rho1 = vec3::dot( b1, b1 );
    float rho2 = vec3::dot( b2, b2 );
    float rho3 = vec3::dot( b3, b3 );
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
    B = mat3( b1, b2, b3 );
    V = mat3( v1, v2, v3 );
}

__host__ __device__ __forceinline__ void QRGivensQuaternion( float a1, float a2, float &ch, float &sh )
{
    // a1 = pivot point on diagonal
    // a2 = lower triangular entry we want to annihilate
    float rho = sqrtf( a1*a1 + a2*a2 );

    sh = rho > EPSILON ? a2 : 0;
    ch = fabsf(a1) + fmaxf( rho, EPSILON );
    bool b = a1 < 0;
    condSwap( b, sh, ch );
    float w = rsqrtf( ch*ch + sh*sh );

    ch *= w;
    sh *= w;
}

__host__ __device__ void QRDecomposition( const mat3 &B, mat3 &Q, mat3 &R )
{
    R = B;

    // QR decomposition of 3x3 matrices using Givens rotations to
    // eliminate elements B21, B31, B32
    quat qQ; // cumulative rotation
    mat3 U;
    float ch, sh, s0, s1;

    // first givens rotation
    QRGivensQuaternion( R[0], R[1], ch, sh );

    s0 = 1-2*sh*sh;
    s1 = 2*sh*ch;
    U = mat3(  s0, s1, 0,
              -s1, s0, 0,
                0,  0, 1 );

    R = mat3::multiplyAtB( U, R );

    // update cumulative rotation
    qQ = quat( ch*qQ.w-sh*qQ.z, ch*qQ.x+sh*qQ.y, ch*qQ.y-sh*qQ.x, sh*qQ.w+ch*qQ.z );

    // second givens rotation
    QRGivensQuaternion( R[0], R[2], ch, sh );

    s0 = 1-2*sh*sh;
    s1 = 2*sh*ch;
    U = mat3(  s0, 0, s1,
                0, 1,  0,
              -s1, 0, s0 );

    R = mat3::multiplyAtB( U, R );

    // update cumulative rotation
    qQ = quat( ch*qQ.w+sh*qQ.y, ch*qQ.x+sh*qQ.z, ch*qQ.y-sh*qQ.w, ch*qQ.z-sh*qQ.x );

    // third Givens rotation
    QRGivensQuaternion( R[4], R[5], ch, sh );

    s0 = 1-2*sh*sh;
    s1 = 2*sh*ch;
    U = mat3( 1,   0,  0,
              0,  s0, s1,
              0, -s1, s0 );

    R = mat3::multiplyAtB( U, R );

    // update cumulative rotation
    qQ = quat( ch*qQ.w-sh*qQ.x, sh*qQ.w+ch*qQ.x, ch*qQ.y+sh*qQ.z, ch*qQ.z-sh*qQ.y );

    // qQ now contains final rotation for Q
    Q = mat3::fromQuat(qQ);
}

/*
 * McAdams, Selle, Tamstorf, Teran, and Sifakis. Computing the Singular Value Decomposition of 3 x 3
 * matrices with minimal branching and elementary floating point operations
 * Computes SVD of 3x3 matrix A = W * S * V'
 */
__host__ __device__ __forceinline__ void computeSVD( const mat3 &A, mat3 &W, mat3 &S, mat3 &V )
{
    // normal equations matrix
    mat3 ATA = mat3::multiplyAtB( A, A );

/// 2. Symmetric Eigenanlysis
    quat qV;
    jacobiEigenanalysis( ATA, qV );
    V = mat3::fromQuat(qV);
    mat3 B = A * V;

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
__host__ __device__ void computePD( const mat3 &A, mat3 &R )
{
    // U is unitary matrix (i.e. orthogonal/orthonormal)
    // P is positive semidefinite Hermitian matrix
    mat3 W, S, V;
    computeSVD( A, W, S, V );
    R = mat3::multiplyABt( W, V );
}

/*
 * In snow we desire both SVD and polar decompositions simultaneously without
 * re-computing USV for polar.
 * here is a function that returns all the relevant values
 * SVD : A = W * S * V'
 * PD : A = R * E
 */
__host__ __device__ void computeSVDandPD( const mat3 &A, mat3 &W, mat3 &S, mat3 &V, mat3 &R )
{
    computeSVD( A, W, S, V );
    R = mat3::multiplyABt( W, V );
}

#endif // DECOMPOSITION_H
