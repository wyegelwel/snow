/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   implicit.cu
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 26 Apr 2014
**
**************************************************************************/

#ifndef IMPLICIT_H
#define IMPLICIT_H

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_INCLUDE
#include "geometry/grid.h"
#include "sim/particle.h"
#include "cuda/vector.cu"

#include "cuda/atomic.cu"
#include "cuda/decomposition.cu"


__device__ void computedR(mat3 &df, mat3 &Se, mat3 &Re, mat3 &dR){
    mat3 V = mat3::multiplyAtB(Re, dF) - mat3::multiplyAtB(dF, Re);

    // Solve for compontents of R^T * dR
    mat3 S = mat3(S[0]+S[4], S[5], -S[2], //remember, column major
                  S[5], S[0]+S[8], S[1],
                  -S[2], S[1], S[4]+S[8]);


    vec3 b(V[3], V[6], V[7]);

    vec3 x = mat3::inverse(S) * b; // Should replace this with a linear system solver function

    // Fill R^T * dR
    mat3 RTdR = mat3( 0, -x.x, -x.y, //remember, column major
                      x.x, 0, -x.z,
                      x.y, x.z, 0);

    dR = Re*RTdR;
}

__device__ void compute_dJF_invTrans(mat3 &F, mat3 &dF, mat3 &dJF_invTrans){
    mat3 tmp;
    // considering F[0]
    tmp[0] = 0; tmp[3] =  0;    tmp[6] =  0;
    tmp[1] = 0; tmp[4] =  F[8]; tmp[7] = -F[7];
    tmp[2] = 0; tmp[5] = -F[5]; tmp[8] =  Fp[4];
    dJF_invTrans[0] = mat3::innerProduct(tmp, dF);

    // considering F[1]
    tmp[0] =  0;    tmp[3] = 0; tmp[6] =  0;
    tmp[1] = -F[8]; tmp[4] = 0; tmp[7] =  F[6];
    tmp[2] =  F[5]; tmp[5] = 0; tmp[8] = -F[3];
    dJF_invTrans[1] = mat3::innerProduct(tmp, dF);

    // considering F[2]
    tmp[0] =  0;    tmp[3] = 0;     tmp[6] = 0;
    tmp[1] =  F[7]; tmp[4] = -F[6]; tmp[7] = 0;
    tmp[2] = -F[4]; tmp[5] =  F[3]; tmp[8] = 0;
    dJF_invTrans[2] = mat3::innerProduct(tmp, dF);

    // considering F[3]
    tmp[0] =  0; tmp[3] = -F[8]; tmp[6] = F[7];
    tmp[1] =  0; tmp[4] =  0;    tmp[7] = 0;
    tmp[2] =  0; tmp[5] =  F[2]; tmp[8] = -F[1];
    dJF_invTrans[3] = mat3::innerProduct(tmp, dF);

    // considering F[4]
    tmp[0] =  F[8]; tmp[3] = 0; tmp[6] = -F[6];
    tmp[1] =  0;    tmp[4] = 0; tmp[7] =  0;
    tmp[2] = -F[2]; tmp[5] = 0; tmp[8] =  F[0];
    dJF_invTrans[4] = mat3::innerProduct(tmp, dF);

    // considering F[5]
    tmp[0] =  -F[7]; tmp[3] =  F[6]; tmp[6] = 0;
    tmp[1] =   0;    tmp[4] =  0;    tmp[7] = 0;
    tmp[2] =   F[1]; tmp[5] = -F[0]; tmp[8] = 0;
    dJF_invTrans[5] = mat3::innerProduct(tmp, dF);

    // considering F[6]
    tmp[0] =   0; tmp[3] =  F[5]; tmp[6] = -F[4];
    tmp[1] =   0; tmp[4] = -F[2]; tmp[7] = F[1];
    tmp[2] =   0; tmp[5] =  0;    tmp[8] = 0;
    dJF_invTrans[6] = mat3::innerProduct(tmp, dF);

    // considering F[7]
    tmp[0] = -F[5]; tmp[3] = 0; tmp[6] =  F[3];
    tmp[1] =  F[2]; tmp[4] = 0; tmp[7] = -F[0];
    tmp[2] =  0;    tmp[5] = 0; tmp[8] =  0;
    dJF_invTrans[7] = mat3::innerProduct(tmp, dF);

    // considering F[8]
    tmp[0] =  F[4]; tmp[3] = -F[3]; tmp[6] =  0;
    tmp[1] = -F[1]; tmp[4] =  F[0]; tmp[7] =  0;
    tmp[2] =  0;    tmp[5] =  0;    tmp[8] =  0;
    dJF_invTrans[8] = mat3::innerProduct(tmp, dF);
}


// TODO: Replace JFe_invTrans with the trans of adjugate
__device__ void computeAp( Particle &particles, mat3 &dF, mat3 &Ap, ParticleFeHatCache &pFeHatCache )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particles[particleIdx];
    mat3 &dF = dFs[particleIdx];
    mat3 &Ap = Aps[particleIdx];

    mat3 &Fp = particle.plasticF; //for the sake of making the code look like the math
    mat3 &Fe = pFeHatCache.FeHat;

    float Jpp = mat3::determinant(Fp);
    float Jep = mat3::determinant(Fe);

    float muFp = material->mu*__expf(material->xi*(1-Jpp));
    float lambdaFp = material->lambda*__expf(material->xi*(1-Jpp));

    mat3 &Re = pFeHatCache.ReHat;
    mat3 &Se = pFeHatCache.SeHat;

    mat3 dR;
    computedR(dF, Se, Re, dR);

    mat3 jFe_invTrans = Jep*mat3::transpose(mat3::inverse(Fe));

    mat3 dJFe_invTrans;
    compute_dJF_invTrans(Fe, dF, dJFe_invTrans);

    Ap = (2*muFp*(dF - dRe) + lambdaFp*jFe_invTrans*mat3::innerProduct(jFe_invTrans, dF) + lambdaFp*(Jep - 1)*dJFe_invTrans);

//    sigma = (2*muFp*(Fe-Re)*mat3::transpose(Fe)+lambdaFp*(Jep-1)*Jep*mat3(1.0f)) * (particle.volume);
//    sigma = (2*muFp*mat3::multiplyABt(Fe-Re, Fe) + mat3(lambdaFp*(Jep-1)*Jep)) * -particle.volume;
}

__device__ void computedF(Particle &particle, Grid *grid, mat3 &dF){
    const vec3 &pos = particle.position;
    const glm::ivec3 &dim = grid->dim;
    const float h = grid->h;

    // Compute neighborhood of particle in grid
    vec3 gridIndex = (pos - grid->pos) / h,
         gridMax = vec3::floor( gridIndex + vec3(2,2,2) ),
         gridMin = vec3::ceil( gridIndex - vec3(2,2,2) );
    glm::ivec3 maxIndex = glm::clamp( VEC2IVEC(gridMax), glm::ivec3(0,0,0), dim ),
               minIndex = glm::clamp( VEC2IVEC(gridMin), glm::ivec3(0,0,0), dim );

    // Fill dF
    int rowSize = dim.z+1;
    int pageSize = (dim.y+1)*rowSize;
    for ( int i = minIndex.x; i <= maxIndex.x; ++i ) {
        vec3 d, s;
        d.x = gridIndex.x - i;
        d.x *= ( s.x = ( d.x < 0 ) ? -1.f : 1.f );
        int pageOffset = i*pageSize;
        for ( int j = minIndex.y; j <= maxIndex.y; ++j ) {
            d.y = gridIndex.y - j;
            d.y *= ( s.y = ( d.y < 0 ) ? -1.f : 1.f );
            int rowOffset = pageOffset + j*rowSize;
            for ( int k = minIndex.z; k <= maxIndex.z; ++k ) {
                d.z = gridIndex.z - k;
                d.z *= ( s.z = ( d.z < 0 ) ? -1.f : 1.f );
                vec3 du_j = dt * u[rowOffset+k];

                float w;
                vec3 wg;
                weightAndGradient( -s, d, w, wg );

                dF += mat3::outerProduct(du_j, wg);
            }
        }
    }

    dF *= particle.elasticF;
}

/**
 * Called over particles over nodes the particle affects. (numParticles * 64)
 *
 * Recommended:
 *  dim3 blockDim = dim3(numParticles / threadCount, 64);
 *  dim3 threadDim = dim3(threadCount/64, 64);
 *
 **/
__global__ void computedFandAps( Particle *particles, Grid *grid, float dt, ParticleFeHatCache *particleFeHatCache, vec3 *u, mat3 *Aps)
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particles[particleIdx];
    mat3 dF(0.0f);

    computedF(particle, grid, dF);


    mat3 &Ap = Aps[particleIdx];
    ParticleFeHatCache &pFeHatCache = particleFeHatCache[particleIdx];

    computeAp(particle, dF, Ap, pFeHatCache);
 }





/**
 * Computes the matrix-vector product Eu. All the pointer arguments are assumed to be
 * device pointers.
 *
 *      u:  vector to multiply
 * result:  store the values of Eu
 */
__host__ void computeEu( Particle *particles, int numParticles, Grid *grid, float dt, vec3 *u, vec3 *result )
{

    static const int threadCount = 128;
    dim3 blocks = dim3( numParticles/threadCount, 64 );
    dim3 threads = dim3( threadCount/64, 64 );



}

#endif // IMPLICIT_H
