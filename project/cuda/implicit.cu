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
#include <helper_functions.h>
#include <helper_cuda.h>

#define CUDA_INCLUDE
#include "geometry/grid.h"
#include "sim/material.h"
#include "sim/particle.h"
#include "sim/particlegridnode.h"
#include "cuda/vector.cu"

#include "cuda/atomic.cu"
#include "cuda/blas.cu"
#include "cuda/caches.h"
#include "cuda/decomposition.cu"
#include "cuda/weighting.cu"

#define BETA 0.5
#define MAX_ITERATIONS 30
#define STOPPING_EPSILON 1e-8

/**
 * Called over particles
 **/
#define VEC2IVEC( V ) ( glm::ivec3((int)V.x, (int)V.y, (int)V.z) )
__global__ void computedF(Particle *particles, Grid *grid, float dt, ParticleGridNode *nodes, vec3 *dus, Implicit::ParticleCache *pCaches ){
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particles[particleIdx];
    Implicit::ParticleCache &pCache = pCaches[particleIdx];
    mat3 dF(0.0f);

    const vec3 &pos = particle.position;
    const glm::ivec3 &dim = grid->dim;
    const float h = grid->h;

    // Compute neighborhood of particle in grid
    vec3 gridIndex = (pos - grid->pos) / h,
         gridMax = vec3::floor( gridIndex + vec3(2,2,2) ),
         gridMin = vec3::ceil( gridIndex - vec3(2,2,2) );
    glm::ivec3 maxIndex = glm::clamp( VEC2IVEC(gridMax), glm::ivec3(0,0,0), dim ),
               minIndex = glm::clamp( VEC2IVEC(gridMin), glm::ivec3(0,0,0), dim );

    mat3 vGradient(0.0f);

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
                vec3 wg;
                weightGradient( -s, d, wg );

                vec3 du_j = dt * dus[rowOffset+k];
                dF += mat3::outerProduct(du_j, wg);

                vGradient += mat3::outerProduct(dt*nodes[rowOffset+k].velocity, wg);

            }
        }
    }

    pCache.dF = dF * particle.elasticF;

    pCache.FeHat = mat3::addIdentity(vGradient) * particle.elasticF;
    computePD( pCache.FeHat, pCache.ReHat, pCache.SeHat );
}

/** Currently computed in computedF, we could parallelize this and computedF but not sure what the time benefit would be*/
//__global__ void computeFeHat(Particle *particles, Grid *grid, float dt, ParticleGridNode *nodes, ACache *ACaches){
//    int particleIdx = blockIdx.x*blockDim.x + threadIdx.x;

//       Particle &particle = particles[particleIdx];
//       ACache &ACache = ACaches[particleIdx];

//       vec3 particleGridPos = (particle.position - grid->pos) / grid->h;
//       glm::ivec3 min = glm::ivec3(std::ceil(particleGridPos.x - 2), std::ceil(particleGridPos.y - 2), std::ceil(particleGridPos.z - 2));
//       glm::ivec3 max = glm::ivec3(std::floor(particleGridPos.x + 2), std::floor(particleGridPos.y + 2), std::floor(particleGridPos.z + 2));

//       mat3 vGradient(0.0f);

//       // Apply particles contribution of mass, velocity and force to surrounding nodes
//       min = glm::max(glm::ivec3(0.0f), min);
//       max = glm::min(grid->dim, max);
//       for (int i = min.x; i <= max.x; i++){
//           for (int j = min.y; j <= max.y; j++){
//               for (int k = min.z; k <= max.z; k++){
//                   int currIdx = grid->getGridIndex(i, j, k, grid->dim+1);
//                   ParticleGridNode &node = nodes[currIdx];

//                   vec3 wg;
//                   weightGradient(particleGridPos - vec3(i, j, k), wg);

//                   vGradient += mat3::outerProduct(dt*node.velocity, wg);
//               }
//           }
//       }

//       ACache.FeHat = mat3::addIdentity(vGradient) * particle.elasticF;
//       computePD(ACache.FeHat, ACache.ReHat, ACache.SeHat);
//}

__device__ void computedR(mat3 &dF, mat3 &Se, mat3 &Re, mat3 &dR){
    mat3 V = mat3::multiplyAtB(Re, dF) - mat3::multiplyAtB(dF, Re);

    // Solve for compontents of R^T * dR
    mat3 A = mat3(S[0]+S[4], S[5], -S[2], //remember, column major
                  S[5], S[0]+S[8], S[1],
                  -S[2], S[1], S[4]+S[8]);

    vec3 b(V[3], V[6], V[7]);
    vec3 x = mat3::solve(A, b);// Should replace this with a linear system solver function

    // Fill R^T * dR
    mat3 RTdR = mat3( 0, -x.x, -x.y, //remember, column major
                      x.x, 0, -x.z,
                      x.y, x.z, 0);

    dR = Re*RTdR;
}

/**
 * This function involves taking the partial derivative of the adjugate of F
 * with respect to each element of F. This process results in a 3x3 block matrix
 * where each block is the 3x3 partial derivative for an element of F
 *
 * Let F = [ a b c
 *           d e f
 *           g h i ]
 *
 * Let adjugate(F) = [ ei-hf  hc-bi  bf-ec
 *                     gf-di  ai-gc  dc-af
 *                     dh-ge  gb-ah  ae-db ]
 *
 * Then d/da (adjugate(F) = [ 0   0   0
 *                            0   i  -f
 *                            0  -h   e ]
 *
 * The other 8 partials will have similar form. See (and run) the code in
 * matlab/derivateAdjugateF.m for the full computation as well as to see where
 * these seemingly magic values came from.
 *
 *
 */
__device__ void compute_dJF_invTrans(mat3 &F, mat3 &dF, mat3 &dJF_invTrans){
    dJF_invTrans[0] = F[4]*dF[8] - F[5]*dF[5] + F[8]*dF[4] - F[7]*dF[7];
    dJF_invTrans[1] = F[5]*dF[2] - F[8]*dF[1] - F[3]*dF[8] + F[6]*dF[7];
    dJF_invTrans[2] = F[3]*dF[5] - F[4]*dF[2] + F[7]*dF[1] - F[6]*dF[4];
    dJF_invTrans[3] = F[2]*dF[5] - F[1]*dF[8] - F[8]*dF[3] + F[7]*dF[6];
    dJF_invTrans[4] = F[0]*dF[8] - F[2]*dF[2] + F[8]*dF[0] - F[6]*dF[6];
    dJF_invTrans[5] = F[1]*dF[2] - F[0]*dF[5] - F[7]*dF[0] + F[6]*dF[3];
    dJF_invTrans[6] = F[1]*dF[7] - F[2]*dF[4] + F[5]*dF[3] - F[4]*dF[6];
    dJF_invTrans[7] = F[2]*dF[1] - F[5]*dF[0] - F[0]*dF[7] + F[3]*dF[6];
    dJF_invTrans[8] = F[0]*dF[4] - F[1]*dF[1] + F[4]*dF[0] - F[3]*dF[3];
}

/**
 * Called over particles
 **/
// TODO: Replace JFe_invTrans with the trans of adjugate
__global__ void computeAp( Particle *particles, MaterialConstants *material, Implicit::ParticleCache *pCaches )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    Particle &particle = particles[particleIdx];
    Implicit::ParticleCache &pCache = pCaches[particleIdx];
    mat3 dF = pCache.dF;

    mat3 &Fp = particle.plasticF; //for the sake of making the code look like the math
    mat3 &Fe = pCache.FeHat;

    float Jpp = mat3::determinant(Fp);
    float Jep = mat3::determinant(Fe);

    float muFp = material->mu*__expf(material->xi*(1-Jpp));
    float lambdaFp = material->lambda*__expf(material->xi*(1-Jpp));

    mat3 &Re = pCache.ReHat;
    mat3 &Se = pCache.SeHat;

    mat3 dR;
    computedR(dF, Se, Re, dR);

    mat3 dJFe_invTrans;
    compute_dJF_invTrans(Fe, dF, dJFe_invTrans);

    mat3 jFe_invTrans = Jep * mat3::transpose(mat3::inverse(Fe));

    pCache.Ap = (2*muFp*(dF - dR) + lambdaFp*jFe_invTrans*mat3::innerProduct(jFe_invTrans, dF) + lambdaFp*(Jep - 1)*dJFe_invTrans);
}


__global__ void computedf( Particle *particles, Grid *grid, Implicit::ParticleCache *pCache, vec3 *df )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particles[particleIdx];
    vec3 gridPos = (particle.position-grid->pos)/grid->h;

    glm::ivec3 ijk;
    Grid::gridIndexToIJK( threadIdx.y, glm::ivec3(4,4,4), ijk );
    ijk += glm::ivec3( gridPos.x-1, gridPos.y-1, gridPos.z-1 );

    if ( Grid::withinBoundsInclusive(ijk, glm::ivec3(0,0,0), grid->dim) ) {

        vec3 wg;
        vec3 nodePos(ijk);
        weightGradient( gridPos-nodePos, wg );
        vec3 df_j = -particle.volume * mat3::multiplyABt( pCache[particleIdx].Ap, particle.elasticF ) * wg;

        atomicAdd( &df[Grid::getGridIndex(ijk,grid->nodeDim())], df_j );
    }
}

__global__ void computeEuResult( ParticleGridNode *nodes, int numNodes, float dt, vec3 *u, vec3 *df, vec3 *result )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    result[tid] = u[tid] - (BETA*dt/nodes[tid].mass)*df[tid];
}

/**
 * Computes the matrix-vector product Eu. All the pointer arguments are assumed to be
 * device pointers.
 */
__host__ void computeEu( const Particle *particles, int numParticles,
                         const Grid *grid, const ParticleGridNode *nodes, int numNodes,
                         float dt, const vec3 *u, const vec3 *df, vec3 *result, Implicit::ParticleCache *pCache )
{
    static const int threadCount = 128;

    dim3 blocks = dim3( numParticles/threadCount, 64 );
    dim3 threads = dim3( threadCount/64, 64 );

    computedf<<< blocks, threads >>>( particles, grid, pCache, df );
    checkCudaErrors( cudaDeviceSynchronize() );
    computeEuResult<<< numNodes/threadCount, threadCount >>>( nodes, numNodes, dt, u, df, result );
    checkCudaErrors( cudaDeviceSynchronize() );
}

__global__ void initializeVelocities( ParticleGridNode *nodes, int numNodes, vec3 *v )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    v[tid] = nodes[tid].velocity;
}

__global__ void initializeRP( int numNodes, const vec3 *vstar, const vec3 *Ev0, vec3 *r0, vec3 *p0 )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    r0[tid] = vstar[tid] - Ev0[tid];
    p0[tid] = r0[tid];
}

__global__ void initializeQ( int numNodes, vec3 *s0, vec3 *q0 )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    q0[tid] = s0[tid];
}



__host__ void initializeConjugateResidual( Particle *particles, int numParticles,
                                           Grid *grid, ParticleGridNode *nodes, int numNodes,
                                           float dt, Implicit::NodeCache *nodeCache, float &gamma, float &alpha, Implicit::ParticleCache *pCache )
{
    static const int threadCount = 128;
    static const dim3 blocks( numNodes/threadCount );
    static const dim3 threads( threadCount );

    initializeVelocities<<< blocks, threads >>>( nodes, numNodes, nodeCache->v );
    checkCudaErrors( cudaDeviceSynchronize() );

    computeEu( particles, numParticles, grid, nodes, numNodes, dt, nodeCache->v, nodeCache->df, nodeCache->r, pCache );
    initializeRP<<< blocks, threads >>>( numNodes, nodeCache->v, nodeCache->r, nodeCache->r, nodeCache->p );
    checkCudaErrors( cudaDeviceSynchronize() );

    computeEu( particles, numParticles, grid, nodes, numNodes, dt, nodeCache->r, nodeCache->df, nodeCache->s, pCache );
    initializeQ<<< blocks, threads >>>( numNodes, nodeCache->s, nodeCache->q );
    checkCudaErrors( cudaDeviceSynchronize() );

    gamma = innerProduct( numNodes, nodeCache->r, nodeCache->s, nodeCache->innerProduct );
    alpha = gamma / innerProduct( numNodes, nodeCache->q, nodeCache->q, nodeCache->innerProduct );
}

__global__ void finishConjugateResidualKernel( ParticleGridNode *nodes, int numNodes, const vec3 *v )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    nodes[tid].velocity = v[tid];
}

__host__ void finishConjugateResidual( ParticleGridNode *nodes, int numNodes, const vec3 *v )
{
    finishConjugateResidualKernel<<< (numNodes+255)/256, 256 >>>( nodes, numNodes, v );
    cudaDeviceSynchronize();
}

__host__ void computeNodeVelocitiesImplicit( Particle *particles, int numParticles,
                                             Grid *grid, ParticleGridNode *nodes, int numNodes,
                                             float dt, Implicit::NodeCache *nodeCache, Implicit::ParticleCache *pCache )
{
    float gamma, alpha, beta;
    initializeConjugateResidual( particles, numParticles, grid, nodes, numNodes, dt, nodeCache, gamma, alpha, pCache );

    int k = 0;
    float d;
    do {
        scaleAndAdd( numNodes, 1.f, nodeCache->v, alpha, nodeCache->p, nodeCache->v );
        scaleAndAdd( numNodes, 1.f, nodeCache->r, -alpha, nodeCache->q, nodeCache->r );
        computeEu( particles, numParticles, grid, nodes, numNodes, dt, nodeCache->r, nodeCache->df, nodeCache->s, pCache );
        beta = innerProduct( numNodes, nodeCache->r, nodeCache->s, nodeCache->innerProduct ) / gamma;
        gamma = beta * gamma;
        scaleAndAdd( numNodes, 1.f, nodeCache->r, beta, nodeCache->p, nodeCache->p );
        scaleAndAdd( numNodes, 1.f, nodeCache->s, beta, nodeCache->q, nodeCache->q );
        alpha = gamma / innerProduct( numNodes, nodeCache->q, nodeCache->q, nodeCache->innerProduct );
        d = alpha * alpha * innerProduct( numNodes, nodeCache->p, nodeCache->p, nodeCache->innerProduct ) / numNodes;
    } while ( k < MAX_ITERATIONS && d > STOPPING_EPSILON );

    finishConjugateResidual( nodes, numNodes, nodeCache->v );
}



#endif // IMPLICIT_H
