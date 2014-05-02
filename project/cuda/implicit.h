/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   implicit.h
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
#include "math.h"

#define CUDA_INCLUDE
#include "geometry/grid.h"
#include "sim/caches.h"
#include "sim/material.h"
#include "sim/particle.h"
#include "sim/particlegridnode.h"
#include "cuda/vector.h"

#include "cuda/helpers.h"
#include "cuda/atomic.h"
#include "cuda/decomposition.h"
#include "cuda/weighting.h"

#include "common/common.h"

#define BETA 0.5f
#define MAX_ITERATIONS 15
#define RESIDUAL_THRESHOLD 1e-20

/**
 * Called over particles
 **/
__global__ void computedF( const Particle *particles, ParticleCache *particleCache, int numParticles,
                           const Grid *grid, const NodeCache *nodeCaches,
                           NodeCache::Offset uOffset, float dt )
{
    int particleIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if ( particleIdx >= numParticles ) return;

    const Particle &particle = particles[particleIdx];

    const glm::ivec3 &dim = grid->dim;

    // Compute neighborhood of particle in grid
    vec3 gridIndex = (particle.position - grid->pos) / grid->h,
         gridMax = vec3::floor( gridIndex + vec3(2,2,2) ),
         gridMin = vec3::ceil( gridIndex - vec3(2,2,2) );
    glm::ivec3 maxIndex = glm::clamp( glm::ivec3(gridMax), glm::ivec3(0,0,0), dim ),
               minIndex = glm::clamp( glm::ivec3(gridMin), glm::ivec3(0,0,0), dim );

    // Fill dF
    mat3 dF(0.0f);
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
                weightGradient( s, d, wg );

                const NodeCache &nodeCache = nodeCaches[rowOffset+k];
                vec3 du_j = dt * nodeCache[uOffset];

                dF += mat3::outerProduct( du_j, wg );
            }
        }
    }

    particleCache->dFs[particleIdx] = dF * particle.elasticF;
}

/** Currently computed in computedF, we could parallelize this and computedF but not sure what the time benefit would be*/
__global__ void computeFeHat( Particle *particles, ParticleCache *particleCache, int numParticles, Grid *grid, float dt, Node *nodes )
{
    int particleIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if ( particleIdx >= numParticles ) return;

    Particle &particle = particles[particleIdx];

    vec3 particleGridPos = (particle.position - grid->pos) / grid->h;
    glm::ivec3 min = glm::ivec3(std::ceil(particleGridPos.x - 2), std::ceil(particleGridPos.y - 2), std::ceil(particleGridPos.z - 2));
    glm::ivec3 max = glm::ivec3(std::floor(particleGridPos.x + 2), std::floor(particleGridPos.y + 2), std::floor(particleGridPos.z + 2));

    mat3 vGradient(0.0f);

    min = glm::max(glm::ivec3(0.0f), min);
    max = glm::min(grid->dim, max);
    for (int i = min.x; i <= max.x; i++){
        for (int j = min.y; j <= max.y; j++){
            for (int k = min.z; k <= max.z; k++){
                int currIdx = grid->getGridIndex(i, j, k, grid->dim+1);
                Node &node = nodes[currIdx];

                vec3 wg;
                weightGradient(particleGridPos - vec3(i, j, k), wg);

                vGradient += mat3::outerProduct(dt*node.velocity, wg);
            }
        }
    }

    mat3 &FeHat = particleCache->FeHats[particleIdx];
    mat3 &ReHat = particleCache->ReHats[particleIdx];
    mat3 &SeHat = particleCache->SeHats[particleIdx];

    FeHat = mat3::addIdentity(vGradient) * particle.elasticF;
    computePD( FeHat, ReHat, SeHat );
}

/**
 * Computes dR
 *
 * FeHat = Re * Se (polar decomposition)
 *
 * Re is assumed to be orthogonal
 * Se is assumed to be symmetry Positive semi definite
 *
 *
 */
__device__ void computedR( const mat3 &dF, const mat3 &Se, const mat3 &Re, mat3 &dR )
{
    mat3 V = mat3::multiplyAtB( Re, dF ) - mat3::multiplyAtB( dF, Re );

    // Solve for compontents of R^T * dR
    mat3 A = mat3( Se[0]+Se[4],       Se[5],      -Se[2], //remember, column major
                         Se[5], Se[0]+Se[8],       Se[1],
                        -Se[2],       Se[1], Se[4]+Se[8] );

    vec3 b( V[3], V[6], V[7] );
    vec3 x = mat3::solve( A, b ); // Should replace this with a linear system solver function

    // Fill R^T * dR
    mat3 RTdR = mat3(   0, -x.x, -x.y, //remember, column major
                      x.x,    0, -x.z,
                      x.y,  x.z,    0 );

    dR = Re*RTdR;
}

/**
 * This function involves taking the partial derivative of the cofactor of F
 * with respect to each element of F. This process results in a 3x3 block matrix
 * where each block is the 3x3 partial derivative for an element of F
 *
 * Let F = [ a b c
 *           d e f
 *           g h i ]
 *
 * Let cofactor(F) = [ ei-hf  gf-di  dh-ge
 *                     hc-bi  ai-gc  gb-ah
 *                     bf-ec  dc-af  ae-db ]
 *
 * Then d/da (cofactor(F) = [ 0   0   0
 *                            0   i  -h
 *                            0  -f   e ]
 *
 * The other 8 partials will have similar form. See (and run) the code in
 * matlab/derivateAdjugateF.m for the full computation as well as to see where
 * these seemingly magic values came from.
 *
 *
 */
__device__ void compute_dJF_invTrans( const mat3 &F, const mat3 &dF, mat3 &dJF_invTrans )
{
    dJF_invTrans[0] = F[4]*dF[8] - F[5]*dF[7] - F[7]*dF[5] + F[8]*dF[4];
    dJF_invTrans[1] = F[5]*dF[6] - F[3]*dF[8] + F[6]*dF[5] - F[8]*dF[3];
    dJF_invTrans[2] = F[3]*dF[7] - F[4]*dF[6] - F[6]*dF[4] + F[7]*dF[3];
    dJF_invTrans[3] = F[2]*dF[7] - F[1]*dF[8] + F[7]*dF[2] - F[8]*dF[1];
    dJF_invTrans[4] = F[0]*dF[8] - F[2]*dF[6] - F[6]*dF[2] + F[8]*dF[0];
    dJF_invTrans[5] = F[1]*dF[6] - F[0]*dF[7] + F[6]*dF[1] - F[7]*dF[0];
    dJF_invTrans[6] = F[1]*dF[5] - F[2]*dF[4] - F[4]*dF[2] + F[5]*dF[1];
    dJF_invTrans[7] = F[2]*dF[3] - F[0]*dF[5] + F[3]*dF[2] - F[5]*dF[0];
    dJF_invTrans[8] = F[0]*dF[4] - F[1]*dF[3] - F[3]*dF[1] + F[4]*dF[0];
}

/**
 * Called over particles
 **/
__global__ void computeAp( const Particle *particles, ParticleCache *particleCache, int numParticles )
{
    int particleIdx =  blockIdx.x*blockDim.x + threadIdx.x;
    if ( particleIdx >= numParticles ) return;

    const Particle &particle = particles[particleIdx];
    const Material &material = particle.material;

    const mat3 &dF = particleCache->dFs[particleIdx];

    const mat3 &Fp = particle.plasticF; //for the sake of making the code look like the math
    const mat3 &Fe = particleCache->FeHats[particleIdx];
    const mat3 &Re = particleCache->ReHats[particleIdx];
    const mat3 &Se = particleCache->SeHats[particleIdx];

    float Jpp = mat3::determinant(Fp);
    float Jep = mat3::determinant(Fe);

    float muFp = material.mu*__expf(material.xi*(1-Jpp));
    float lambdaFp = material.lambda*__expf(material.xi*(1-Jpp));

    mat3 dR;
    computedR( dF, Se, Re, dR );

    mat3 dJFe_invTrans;
    compute_dJF_invTrans( Fe, dF, dJFe_invTrans );

    mat3 JFe_invTrans = mat3::cofactor( Fe );

    particleCache->Aps[particleIdx] = (2*muFp*(dF - dR) + lambdaFp*JFe_invTrans*mat3::innerProduct(JFe_invTrans, dF) + lambdaFp*(Jep - 1)*dJFe_invTrans);
}

__global__ void computedf( const Particle *particles, const ParticleCache *particleCache, int numParticles, const Grid *grid, NodeCache *nodeCaches )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    if ( particleIdx >= numParticles ) return;

    const Particle &particle = particles[particleIdx];
    vec3 gridPos = (particle.position-grid->pos)/grid->h;

    glm::ivec3 ijk;
    Grid::gridIndexToIJK( threadIdx.y, glm::ivec3(4,4,4), ijk );
    ijk += glm::ivec3( gridPos-1 );

    if ( Grid::withinBoundsInclusive(ijk, glm::ivec3(0,0,0), grid->dim) ) {

        vec3 wg;
        vec3 nodePos(ijk);
        weightGradient( gridPos-nodePos, wg );
        vec3 df_j = -particle.volume * mat3::multiplyABt( particleCache->Aps[particleIdx], particle.elasticF ) * wg;

        int gridIndex = Grid::getGridIndex( ijk, grid->nodeDim() );
        atomicAdd( &(nodeCaches[gridIndex].df), df_j );
    }
}

__global__ void computeEuResult( const Node *nodes, NodeCache *nodeCaches, int numNodes, float dt, NodeCache::Offset uOffset, NodeCache::Offset resultOffset )
{
    int nodeIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if ( nodeIdx >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[nodeIdx];
    float mass = nodes[nodeIdx].mass;
    float scale = ( mass > 0.f ) ? 1.f/mass : 0.f;
    nodeCache[resultOffset] = nodeCache[uOffset] - BETA*dt*scale*nodeCache.df;
}

__global__ void zero_df( NodeCache *nodeCaches, int numNodes )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    nodeCaches[tid].df = vec3(0.0f);
}

/**
 * Computes the matrix-vector product Eu.
 */
__host__ void computeEu( const Particle *particles, ParticleCache *particleCache, int numParticles,
                         const Grid *grid, const Node *nodes, NodeCache *nodeCaches, int numNodes,
                         NodeCache::Offset uOffset, NodeCache::Offset resultOffset, float dt )
{

    const dim3 pBlocks1D( (numParticles+THREAD_COUNT-1)/THREAD_COUNT );
    const dim3 nBlocks1D( (numNodes+THREAD_COUNT-1)/THREAD_COUNT );
    static const dim3 threads1D( THREAD_COUNT );
    const dim3 pBlocks2D( (numParticles+THREAD_COUNT-1)/THREAD_COUNT, 64 );
    static const dim3 threads2D( THREAD_COUNT / 64, 64 );

    LAUNCH( computedF<<<pBlocks1D,threads1D>>>(particles,particleCache,numParticles,grid,nodeCaches,uOffset,dt) );

    LAUNCH( computeAp<<<pBlocks1D,threads1D>>>(particles,particleCache,numParticles) );

    LAUNCH( zero_df<<<nBlocks1D,threads1D>>>(nodeCaches,numNodes) );

    LAUNCH( computedf<<<pBlocks2D,threads2D>>>(particles,particleCache,numParticles,grid,nodeCaches) );

    LAUNCH( computeEuResult<<<nBlocks1D,threads1D>>>(nodes,nodeCaches,numNodes,dt,uOffset,resultOffset) );
}

__global__ void initializeVKernel( const Node *nodes, NodeCache *nodeCaches, int numNodes )
{
    int nodeIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if ( nodeIdx >= numNodes ) return;
    nodeCaches[nodeIdx].v = nodes[nodeIdx].velocity;
}

__global__ void initializeRPKernel( NodeCache *nodeCaches, int numNodes )
{
    int nodeIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if ( nodeIdx >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[nodeIdx];
    nodeCache.r = nodeCache.v - nodeCache.r;
    nodeCache.p = nodeCache.r;
}

__global__ void initializeApKernel( NodeCache *nodeCaches, int numNodes )
{
    int nodeIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if ( nodeIdx >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[nodeIdx];
    nodeCache.Ap = nodeCache.Ar;
}

__global__ void updateVRKernel( NodeCache *nodeCaches, int numNodes, double alpha )
{
    int nodeIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if ( nodeIdx >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[nodeIdx];
    nodeCache.v += alpha*nodeCache.p;
    nodeCache.r -= alpha*nodeCache.Ap;
}

__global__ void updatePApResidualKernel( NodeCache *nodeCaches, int numNodes, double beta )
{
    int nodeIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if ( nodeIdx >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[nodeIdx];
    nodeCache.p = nodeCache.r + beta * nodeCache.p;
    nodeCache.Ap = nodeCache.Ar + beta * nodeCache.Ap;
    nodeCache.scratch = (double)vec3::dot( nodeCache.r, nodeCache.r );
}

__global__ void finishConjugateResidualKernel( Node *nodes, const NodeCache *nodeCaches, int numNodes )
{
    int nodeIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if ( nodeIdx >= numNodes ) return;
    nodes[nodeIdx].velocity = nodeCaches[nodeIdx].v;
    // Update the velocity change. It is assumed to be set as the pre-update velocity
    nodes[nodeIdx].velocityChange = nodes[nodeIdx].velocity - nodes[nodeIdx].velocityChange;
}

__global__ void scratchReduceKernel( NodeCache *nodeCaches, int numNodes, int reductionSize )
{
    int nodeIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if ( nodeIdx >= numNodes || nodeIdx+reductionSize >= numNodes ) return;
    nodeCaches[nodeIdx].scratch += nodeCaches[nodeIdx+reductionSize].scratch;
}

__host__ double scratchSum( NodeCache *nodeCaches, int numNodes )
{
    static const dim3 blocks( (numNodes+THREAD_COUNT-1)/THREAD_COUNT );
    static const dim3 threads( THREAD_COUNT );
    int steps = (int)(ceilf(log2f(numNodes)));
    int reductionSize = 1 << (steps-1);
    for ( int i = 0; i < steps; i++ ) {
        scratchReduceKernel<<< blocks, threads >>>( nodeCaches, numNodes, reductionSize );
        reductionSize /= 2;
        cudaDeviceSynchronize();
    }
    double result;
    cudaMemcpy( &result, &(nodeCaches[0].scratch), sizeof(double), cudaMemcpyDeviceToHost );
    return result;
}

__global__ void innerProductKernel( NodeCache *nodeCaches, int numNodes, NodeCache::Offset uOffset, NodeCache::Offset vOffset )
{
    int nodeIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if ( nodeIdx >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[nodeIdx];
    nodeCache.scratch = (double)vec3::dot( nodeCache[uOffset], nodeCache[vOffset] );
}

__host__ double innerProduct( NodeCache *nodeCaches, int numNodes, NodeCache::Offset uOffset, NodeCache::Offset vOffset )
{
    const dim3 blocks( (numNodes+THREAD_COUNT-1)/THREAD_COUNT );
    static const dim3 threads( THREAD_COUNT );
    LAUNCH( innerProductKernel<<< blocks, threads >>>(nodeCaches, numNodes, uOffset, vOffset) );
    return scratchSum( nodeCaches, numNodes );
}

__host__ void integrateNodeForces( Particle *particles, ParticleCache *particleCache, int numParticles,
                                   Grid *grid, Node *nodes, NodeCache *nodeCaches, int numNodes,
                                   float dt )
{
    const dim3 blocks( (numNodes+THREAD_COUNT-1)/THREAD_COUNT );
    static const dim3 threads( THREAD_COUNT );

    // No need to sync because it can run in parallel with other kernels
    computeFeHat<<< (numParticles+THREAD_COUNT-1)/THREAD_COUNT, THREAD_COUNT >>>(particles,particleCache,numParticles,grid,dt,nodes);

    // Initialize conjugate residual method
    LAUNCH( initializeVKernel<<<blocks,threads>>>(nodes, nodeCaches, numNodes) );
    computeEu( particles, particleCache, numParticles, grid, nodes, nodeCaches, numNodes, NodeCache::V, NodeCache::R, dt );
    LAUNCH( initializeRPKernel<<<blocks,threads>>>(nodeCaches, numNodes) );
    computeEu( particles, particleCache, numParticles, grid, nodes, nodeCaches, numNodes, NodeCache::R, NodeCache::AR, dt );
    LAUNCH( initializeApKernel<<<blocks,threads>>>(nodeCaches, numNodes) );

    int k = 0;
    float residual;
    do {

        double alphaNum = innerProduct( nodeCaches, numNodes, NodeCache::R, NodeCache::AR );
        double alphaDen = innerProduct( nodeCaches, numNodes, NodeCache::AP, NodeCache::AP );
        double alpha = ( fabsf(alphaDen) > 0.f ) ? alphaNum/alphaDen : 0.f;

        double betaDen = alphaNum;
        LAUNCH( updateVRKernel<<<blocks,threads>>>( nodeCaches, numNodes, alpha ) );
        computeEu( particles, particleCache, numParticles, grid, nodes, nodeCaches, numNodes, NodeCache::R, NodeCache::AR, dt );
        double betaNum = innerProduct( nodeCaches, numNodes, NodeCache::R, NodeCache::AR );
        double beta = ( fabsf(betaDen) > 0.f ) ? betaNum/betaDen : 0.f;

        LAUNCH( updatePApResidualKernel<<<blocks,threads>>>(nodeCaches,numNodes,beta) );
        residual = scratchSum( nodeCaches, numNodes );

        LOG( "k = %3d, rAr = %10g, alpha = %10g, beta = %10g, r = %g", k, alphaNum, alpha, beta, residual );

    } while ( ++k < MAX_ITERATIONS && residual > RESIDUAL_THRESHOLD );

    LAUNCH( finishConjugateResidualKernel<<<blocks,threads>>>(nodes, nodeCaches, numNodes) );
}

#endif // IMPLICIT_H
