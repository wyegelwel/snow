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

#include "cuda/matrix.cu"
#include "cuda/vector.cu"
#include "cuda/quaternion.cu"

#define GAMMA 5.828427124 // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define CSTAR 0.923879532 // cos(pi/8)
#define SSTAR 0.3826834323 // sin(p/8)

__host__ __device__ void jacobiConjugation( int p, int q, mat3 &S, quat &qV )
{
    // eliminate off-diagonal entries Spq, Sqp
    float ch = 2.f * (S[0]-S[4]), ch2 = ch*ch;
    float sh = S[3], sh2 = sh*sh;
    bool flag = ( GAMMA * sh2 < ch2 );
    float w = rsqrt( ch2 + sh2 );
    ch = flag ? w*ch : CSTAR; ch2 = ch*ch;
    sh = flag ? w*sh : SSTAR; sh2 = sh*sh;

    // build rotation matrix Q
    float scale = ch*ch + sh*sh;
    float a = (ch2-sh2) / scale;
    float b = (2.f*sh*ch) / scale;
    mat3 Q( a, b, 0, -b, a, 0, 0, 0, 1 );

    // perform the conjugation to annihilate S = Q' S Q
    S = mat3::multiplyTransposeL( Q, S ) * Q;
    vec3 tmp( qV.x, qV.y, qV.z );
    tmp *= sh;
    sh *= qV.w;
    // original
    qV *= ch;

    // terrible hack, this arranges such that for
    // (p,q) = ((0,1),(1,2),(0,2)), n = (0,1,2)
    int n = 2*q-p-2;
    int x = n;
//    int y = (n+1) % 3;
//    int z = (n+2) % 3;
    int y = ( n == 2 ) ? 0 : n+1;
    int z = ( n > 0 ) ? n-1 : 2;
    qV[z] += sh;
    qV.w -= tmp[z];
    qV[x] += tmp[y];
    qV[y] -= tmp[x];

    // re-arrange matrix for next iteration
    S = mat3( S[4], S[5], S[3],
              S[5], S[8], S[2],
              S[3], S[2], S[0] );
}

/*
 * Wrapper function for the first step. Solve symmetric
 * eigenproblem using jacobi iteration. Given a symmetric
 * matrix S, diagonalize it also returns the cumulative
 * rotation as a quaternion.
 */
__host__ __device__ void jacobiEigenanalysis( mat3 &S, quat &qV )
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
__host__ __device__ __forceinline__ void toMat3( const quat &q, mat3 &M )
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
    M[0] = 1.f - 2.f*(qyy+qzz);
    M[1] = 2.f * (qxy+qwz);
    M[2] = 2.f * (qxz-qwy);
    M[3] = 2.f * (qxy-qwz);
    M[4] = 1.f - 2.f*(qxx+qzz);
    M[5] = 2.f * (qyz+qwx);
    M[6] = 2.f * (qxz+qwy);
    M[7] = 2.f * (qyz-qwx);
    M[8] = 1.f - 2.f*(qxx+qyy);
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

__host__ __device__ void sortSingularValues( mat3 &B, mat3 &V )
{
    // used in step 2
    vec3 b1 = B.column(0); vec3 v1 = V.column(0);
    vec3 b2 = B.column(1); vec3 v2 = V.column(1);
    vec3 b3 = B.column(2); vec3 v3 = V.column(2);
    float rho1 = vec3::dot(b1,b1);
    float rho2 = vec3::dot(b2,b2);
    float rho3 = vec3::dot(b3,b3);
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

__host__ __device__ void QRDecomposition( const mat3 &B, mat3 &Q, mat3 &R )
{
    R = B;

    // QR decomposition of 3x3 matrices using Givens rotations to
    // eliminate elements B21, B31, B32
    quat qQ; // cumulative rotation
    quat qU; // each Givens rotation in quaternion form

    mat3 U;
    float ch, sh;

    // first givens rotation
    QRGivensQuaternion( R[0], R[1], ch, sh );
    qU = quat( ch, 0, 0, sh );
    U = mat3::fromQuat(qU);
//    toMat3( qU, U );
    R = mat3::multiplyTransposeL( U, R );

    // update cumulative rotation
    qQ *= qU;

    // second givens rotation
    QRGivensQuaternion( R[0], R[2], ch, sh );
    qU = quat( ch, 0, -sh, 0 );
    U = mat3::fromQuat(qU);
//    toMat3( qU, U );
    R = mat3::multiplyTransposeL( U, R );

    // update cumulative rotation
    qQ *= qU;

    // third Givens rotation
    QRGivensQuaternion( R[4], R[5], ch, sh );
    qU = quat( ch, sh, 0, 0 );
    U = mat3::fromQuat(qU);
//    toMat3( qU, U );
    R = mat3::multiplyTransposeL( U, R );

    // update cumulative rotation
    qQ *= qU;

    // qQ now contains final rotation for Q
    Q = mat3::fromQuat(qQ);
//    toMat3( qQ, Q );
}

/*
 * McAdams, Selle, Tamstorf, Teran, and Sifakis. Computing the Singular Value Decomposition of 3 x 3
 * matrices with minimal branching and elementary floating point operations
 * Computes SVD of 3x3 matrix A = W * S * V'
 */
__host__ __device__ void computeSVD( const mat3 &A, mat3 &W, mat3 &S, mat3 &V )
{
    // normal equations matrix
    mat3 ATA = mat3::multiplyTransposeL( A, A );

/// 2. Symmetric Eigenanlysis
    quat qV;
    jacobiEigenanalysis( ATA, qV );
    V = mat3::fromQuat(qV);
//    toMat3( qV, V );
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
__host__ __device__ void computePD( const mat3 &A, mat3 &U, mat3 &P )
{
    // U is unitary matrix (i.e. orthogonal/orthonormal)
    // P is positive semidefinite Hermitian matrix
    mat3 W, S, V;
    computeSVD( A, W, S, V );
    mat3 Vt = mat3::transpose(V);
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
__host__ __device__ void computeSVDandPD( const mat3 &A, mat3 &W, mat3 &S, mat3 &V, mat3 &U, mat3 &P )
{
    computeSVD( A, W, S, V );
    mat3 Vt = mat3::transpose(V);
    U = W * Vt;
    P = V * S * Vt;
}

#endif // DECOMPOSITION_H
