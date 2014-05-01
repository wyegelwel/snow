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
#define MAX_ITERATIONS 20
#define RESIDUAL_THRESHOLD 0.001

/**
 * Called over particles
 **/
__global__ void computedF( const Particle *particles, ParticleCache *pCaches,
                           const Grid *grid, const NodeCache *nodeCaches,
                           NodeCache::Offset uOffset, float dt )
{
    int particleIdx = blockIdx.x*blockDim.x + threadIdx.x;

    const Particle &particle = particles[particleIdx];
    ParticleCache &pCache = pCaches[particleIdx];

    const glm::ivec3 &dim = grid->dim;

    // Compute neighborhood of particle in grid
    vec3 gridIndex = (particle.position - grid->pos) / grid->h,
         gridMax = vec3::floor( gridIndex + vec3(2,2,2) ),
         gridMin = vec3::ceil( gridIndex - vec3(2,2,2) );
    glm::ivec3 maxIndex = glm::clamp( glm::ivec3(gridMax), glm::ivec3(0,0,0), dim ),
               minIndex = glm::clamp( glm::ivec3(gridMin), glm::ivec3(0,0,0), dim );

    // Fill dF
    mat3 dF( 0.0f );
    int rowSize = dim.z+1;
    int pageSize = (dim.y+1)*rowSize;
    for ( int i = minIndex.x; i <= maxIndex.x; ++i ) {
        vec3 d, s;
        d.x() = gridIndex.x() - i;
        d.x() *= ( s.x() = ( d.x() < 0 ) ? -1.f : 1.f );
        int pageOffset = i*pageSize;
        for ( int j = minIndex.y; j <= maxIndex.y; ++j ) {
            d.x() = gridIndex.x() - j;
            d.x() *= ( s.x() = ( d.x() < 0 ) ? -1.f : 1.f );
            int rowOffset = pageOffset + j*rowSize;
            for ( int k = minIndex.z; k <= maxIndex.z; ++k ) {
                d.z() = gridIndex.z() - k;
                d.z() *= ( s.z() = ( d.z() < 0 ) ? -1.f : 1.f );
                vec3 wg;
                weightGradient( s, d, wg );
                vec3 du_j = dt * nodeCaches[rowOffset+k][uOffset];
                dF += mat3::outerProduct( du_j, wg );
            }
        }
    }

    pCache.dF = dF * particle.elasticF;
}

/** Currently computed in computedF, we could parallelize this and computedF but not sure what the time benefit would be*/
__global__ void computeFeHat( Particle *particles, Grid *grid, float dt, Node *nodes, ParticleCache *pCaches )
{
    int particleIdx = blockIdx.x*blockDim.x + threadIdx.x;

       Particle &particle = particles[particleIdx];
       ParticleCache &pCache = pCaches[particleIdx];

       vec3 particleGridPos = (particle.position - grid->pos) / grid->h;
       glm::ivec3 min = glm::ivec3(std::ceil(particleGridPos.x() - 2), std::ceil(particleGridPos.x() - 2), std::ceil(particleGridPos.z() - 2));
       glm::ivec3 max = glm::ivec3(std::floor(particleGridPos.x() + 2), std::floor(particleGridPos.x() + 2), std::floor(particleGridPos.z() + 2));

       mat3 vGradient(0.0f);

       min = glm::max(glm::ivec3(0.0f), min);
       max = glm::min(grid->dim, max);
       for (int i = min.x; i <= max.x; i++){
           for (int j = min.y; j <= max.y; j++){
               for (int k = min.z; k <= max.z; k++){
                   int currIdx = grid->getGridIndex(i, j, k, grid->dim+1);
                   Node &node = nodes[currIdx];

                   vec3 wg;
                   weightGradient(particleGridPos-vec3(i,j,k), wg);

                   vGradient += mat3::outerProduct(dt*node.velocity, wg);
               }
           }
       }

       pCache.FeHat = mat3::addIdentity(vGradient) * particle.elasticF;
       computePD( pCache.FeHat, pCache.ReHat, pCache.SeHat );
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
    mat3 RTdR = mat3(   0, -x.x(), -x.x(), //remember, column major
                      x.x(),    0, -x.z(),
                      x.x(),  x.z(),    0 );

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
__global__ void computeAp( const Particle *particles, ParticleCache *pCaches )
{
    int particleIdx =  blockIdx.x*blockDim.x + threadIdx.x;

    const Particle &particle = particles[particleIdx];
    const Material &material = particle.material;

    ParticleCache &pCache = pCaches[particleIdx];

    const mat3 dF = pCache.dF;
    const mat3 &Fp = particle.plasticF; //for the sake of making the code look like the math
    const mat3 &Fe = pCache.FeHat;

    float Jpp = mat3::determinant(Fp);
    float Jep = mat3::determinant(Fe);

    float muFp = material.mu*__expf(material.xi*(1-Jpp));
    float lambdaFp = material.lambda*__expf(material.xi*(1-Jpp));

    const mat3 &Re = pCache.ReHat;
    const mat3 &Se = pCache.SeHat;

    mat3 dR;
    computedR( dF, Se, Re, dR );

    mat3 dJFe_invTrans;
    compute_dJF_invTrans( Fe, dF, dJFe_invTrans );

    mat3 JFe_invTrans = mat3::cofactor( Fe );

    pCache.Ap = 2*muFp*(dF-dR) + lambdaFp*JFe_invTrans*mat3::innerProduct(JFe_invTrans,dF) + lambdaFp*(Jep-1)*dJFe_invTrans;
}

__global__ void computedf( const Particle *particles, const ParticleCache *pCaches, const Grid *grid, NodeCache *nodeCaches )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    const Particle &particle = particles[particleIdx];
    vec3 gridPos = (particle.position-grid->pos)/grid->h;

    glm::ivec3 ijk;
    Grid::gridIndexToIJK( threadIdx.y, glm::ivec3(4,4,4), ijk );
    ijk += glm::ivec3( gridPos-1 );

    if ( Grid::withinBoundsInclusive(ijk, glm::ivec3(0,0,0), grid->dim) ) {

        vec3 wg;
        weightGradient( gridPos-vec3(ijk), wg );

        vec3 df_j = -particle.volume * mat3::multiplyABt( pCaches[particleIdx].Ap, particle.elasticF ) * wg;

        int gridIndex = Grid::getGridIndex( ijk, grid->nodeDim() );
        NodeCache &nodeCache = nodeCaches[gridIndex];
        atomicAdd( &nodeCache.df, df_j );

    }
}

__global__ void computeEuResult( const Node *nodes, NodeCache *nodeCaches, int numNodes, float dt, NodeCache::Offset uOffset, NodeCache::Offset resultOffset )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[tid];
    float scale = ( nodes[tid].mass > 0.f ) ? 1.f / nodes[tid].mass : 0.f;
    nodeCache[resultOffset] = nodeCache[uOffset] - BETA*dt*scale*nodeCache.df;
}

/**
 * Computes the matrix-vector product Eu, without having to explicitly build the matrix E.
 */
__host__ void computeEu( const Particle *particles, ParticleCache *pCaches, int numParticles,
                         const Grid *grid, const Node *nodes, NodeCache *nodeCaches, int numNodes,
                         NodeCache::Offset uOffset, NodeCache::Offset resultOffset, float dt )
{
    const dim3 pBlocks1D( (numParticles+THREAD_COUNT-1)/THREAD_COUNT );
    const dim3 nBlocks1D( (numNodes+THREAD_COUNT-1)/THREAD_COUNT );
    static const dim3 threads1D( THREAD_COUNT );
    const dim3 pBlocks2D( (numParticles+THREAD_COUNT-1)/THREAD_COUNT, 64 );
    static const dim3 threads2D( THREAD_COUNT/64, 64 );

    LAUNCH( computedF<<<pBlocks1D,threads1D>>>(particles,pCaches,grid,nodeCaches,uOffset,dt) )
    LAUNCH( computeAp<<<pBlocks1D,threads1D>>>(particles,pCaches) )
    LAUNCH( computedf<<<pBlocks2D,threads2D>>>(particles,pCaches,grid,nodeCaches) );
    LAUNCH( computeEuResult<<<nBlocks1D,threads1D>>>(nodes,nodeCaches,numNodes,dt,uOffset,resultOffset) );
}

__global__ void initializeVKernel( const Node *nodes, NodeCache *nodeCaches, int numNodes )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    nodeCaches[tid].v = nodes[tid].velocity;
//    nodeCaches[tid].v = vec3(0,0,0);
}

__global__ void initializeRPKernel( const Node *nodes, NodeCache *nodeCaches, int numNodes )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[tid];
    nodeCache.r = nodeCache.v - nodeCache.r;
//    nodeCache.r = nodes[tid].velocity - nodeCache.r;
    nodeCache.p = nodeCache.r;
}

__global__ void initializeApKernel( NodeCache *nodeCaches, int numNodes )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[tid];
    nodeCache.Ap = nodeCache.Ar;
}

__global__ void updateVRKernel( NodeCache *nodeCaches, int numNodes, double alpha )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[tid];
    nodeCache.v += alpha * nodeCache.p;
    nodeCache.r -= alpha * nodeCache.Ap;
}

__global__ void updatePApResidualKernel( NodeCache *nodeCaches, int numNodes, double beta )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[tid];
    nodeCache.p = nodeCache.r + beta * nodeCache.p;
    nodeCache.Ap = nodeCache.Ar + beta * nodeCache.Ap;
    nodeCache.scratch = (double)vec3::dot( nodeCache.r, nodeCache.r );
}

__global__ void finishConjugateResidualKernel( Node *nodes, const NodeCache *nodeCaches, int numNodes )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    nodes[tid].velocity = nodeCaches[tid].v;
    // Update the velocity change. It is assumed to be set as the pre-update velocity
    nodes[tid].velocityChange = nodes[tid].velocity - nodes[tid].velocityChange;
}

__global__ void scratchReduceKernel( NodeCache *nodeCaches, int numNodes, int reductionSize )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes || tid+reductionSize >= numNodes ) return;
    nodeCaches[tid].scratch += nodeCaches[tid+reductionSize].scratch;
}

__host__ double scratchSum( NodeCache *nodeCaches, int numNodes )
{
    const dim3 blocks( (numNodes+THREAD_COUNT-1)/THREAD_COUNT );
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
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numNodes ) return;
    NodeCache &nodeCache = nodeCaches[tid];
    nodeCache.scratch = (double)vec3::dot( nodeCache[uOffset], nodeCache[vOffset] );
}

__host__ double innerProduct( NodeCache *nodeCaches, int numNodes, NodeCache::Offset uOffset, NodeCache::Offset vOffset )
{
    const dim3 blocks( (numNodes+THREAD_COUNT-1)/THREAD_COUNT );
    static const dim3 threads( THREAD_COUNT );
    LAUNCH( innerProductKernel<<< blocks, threads >>>(nodeCaches, numNodes, uOffset, vOffset) );
    return scratchSum( nodeCaches, numNodes );
}

__host__ void integrateNodeForces( Particle *particles, ParticleCache *pCaches, int numParticles,
                                   Grid *grid, Node *nodes, NodeCache *nodeCaches, int numNodes,
                                   float dt )
{
    const dim3 blocks( (numNodes+THREAD_COUNT-1)/THREAD_COUNT );
    static const dim3 threads( THREAD_COUNT );

    // No need to sync because it can run in parallel with other kernels
    LAUNCH( computeFeHat<<< (numParticles+THREAD_COUNT-1)/THREAD_COUNT, THREAD_COUNT >>>(particles, grid, dt, nodes, pCaches) );

    // Initialize conjugate residual method
    LAUNCH( initializeVKernel<<<blocks,threads>>>(nodes, nodeCaches, numNodes) );
    computeEu( particles, pCaches, numParticles, grid, nodes, nodeCaches, numNodes, NodeCache::V, NodeCache::R, dt );
    LAUNCH( initializeRPKernel<<<blocks,threads>>>(nodes, nodeCaches, numNodes) );
    computeEu( particles, pCaches, numParticles, grid, nodes, nodeCaches, numNodes, NodeCache::R, NodeCache::AR, dt );
    LAUNCH( initializeApKernel<<<blocks,threads>>>(nodeCaches, numNodes) );
    double gamma = innerProduct( nodeCaches, numNodes, NodeCache::R, NodeCache::AR );

    int k = 0;
    double residual = innerProduct( nodeCaches, numNodes, NodeCache::R, NodeCache::R );

    LOG( "          INITIAL RESIDUAL = %g", residual );

    do {

        double alphaDen = innerProduct( nodeCaches, numNodes, NodeCache::AP, NodeCache::AP );
        double alpha = ( fabs(alphaDen) > 0.0 ) ? gamma / alphaDen : 0.0;

        LAUNCH( updateVRKernel<<<blocks,threads>>>( nodeCaches, numNodes, alpha ) );
        computeEu( particles, pCaches, numParticles, grid, nodes, nodeCaches, numNodes, NodeCache::R, NodeCache::AR, dt );
        double betaNum = innerProduct( nodeCaches, numNodes, NodeCache::R, NodeCache::AR );
        double beta = ( fabs(gamma) > 0.0 ) ? betaNum / gamma : 0.0;

        LAUNCH( updatePApResidualKernel<<<blocks,threads>>>(nodeCaches,numNodes,beta) );
        residual = scratchSum( nodeCaches, numNodes );

//        LOG( "          RESIDUAL = %g", residual );

        LOG( "k = %3d, gamma = %g, alpha = %g, alphaNum = %g alphaDen = %g  beta = %g, r = %g", k, gamma, alpha, gamma, alphaDen, beta, residual );

        if ( gamma < 0.0 ) LOG( "        !!! NEGATIVE GAMMA !!!        ");

        gamma *= beta;

    } while ( ++k < MAX_ITERATIONS && residual > RESIDUAL_THRESHOLD );

    LAUNCH( finishConjugateResidualKernel<<<blocks,threads>>>(nodes,nodeCaches,numNodes) );
}

#endif // IMPLICIT_H
