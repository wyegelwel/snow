/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   simulation.cu
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 17 Apr 2014
**
**************************************************************************/

#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define CUDA_INCLUDE
#include "sim/collider.h"
#include "sim/material.h"
#include "sim/parameters.h"
#include "sim/particle.h"
#include "sim/particlegrid.h"

#include "common/math.h"

#include "cuda/collider.cu"
#include "cuda/decomposition.cu"
#include "cuda/weighting.cu"

#include "cuda/functions.h"

__host__ __device__ __forceinline__
bool withinBoundsInclusive( const float &v, const float &min, const float &max )
{
    return ( v >= min && v <= max );
}

__host__ __device__ __forceinline__
bool withinBoundsInclusive( const glm::ivec3 &v, const glm::ivec3 &min, const glm::ivec3 &max )
{
    return withinBoundsInclusive(v.x, min.x, max.x) && withinBoundsInclusive(v.y, min.y, max.y) && withinBoundsInclusive(v.z, min.z, max.z);
}


__host__ __device__ __forceinline__
void gridIndexToIJK( int idx, int &i, int &j, int &k,const  glm::ivec3 &nodeDim )
{
    i = idx / (nodeDim.y*nodeDim.z);
    idx = idx % (nodeDim.y*nodeDim.z);
    j = idx / nodeDim.z;
    k = idx % nodeDim.z;
}

__host__ __device__  __forceinline__
int getGridIndex( int i, int j, int k, const glm::ivec3 &nodeDim)
{
    return (i*(nodeDim.y*nodeDim.z) + j*(nodeDim.z) + k);
}

__host__ __device__ __forceinline__
void gridIndexToIJK( int idx, const  glm::ivec3 &nodeDim, glm::ivec3 &IJK )
{
    gridIndexToIJK(idx, IJK.x, IJK.y, IJK.z, nodeDim);
}

__host__ __device__ __forceinline__
int getGridIndex( const glm::ivec3 &IJK, const glm::ivec3 &nodeDim )
{
    return getGridIndex(IJK.x, IJK.y, IJK.z, nodeDim);
}

__device__ void computeSigma( Particle &particle, MaterialConstants *material, mat3 &sigma )
{
    mat3 &Fp = particle.plasticF; //for the sake of making the code look like the math
    mat3 &Fe = particle.elasticF;

    float Jpp = mat3::determinant(Fp);
    float Jep = mat3::determinant(Fe);

    mat3 Re;
    computePD(Fe, Re);

    float muFp = material->mu*__expf(material->xi*(1-Jpp));
    float lambdaFp = material->lambda*__expf(material->xi*(1-Jpp));

//    sigma = (2*muFp*(Fe-Re)*mat3::transpose(Fe)+lambdaFp*(Jep-1)*Jep*mat3(1.0f)) * (particle.volume);
    sigma = (2*muFp*mat3::multiplyABt(Fe-Re, Fe) + mat3(lambdaFp*(Jep-1)*Jep)) * particle.volume;
}

__global__ void computeParticleGridTempData ( Particle *particleData, Grid *grid, MaterialConstants *material, ParticleGridTempData *particleGridTempData )
{
    int particleIdx = blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particleData[particleIdx];
    ParticleGridTempData &pgtd = particleGridTempData[particleIdx];

    pgtd.particleGridPos = (particle.position - grid->pos)/grid->h;
    computeSigma(particle, material, pgtd.sigma);
}

__device__ __forceinline__
void atomicAdd( vec3 *add, vec3 toAdd )
{
    atomicAdd(&(add->x), toAdd.x);
    atomicAdd(&(add->y), toAdd.y);
    atomicAdd(&(add->z), toAdd.z);
}

/**
 * Called on each particle.
 *
 * Each particle adds it's mass, velocity and force contribution to the grid nodes within 2h of itself.
 *
 * In:
 * particleData -- list of particles
 * grid -- Stores grid paramters
 * worldParams -- Global parameters dealing with the physics of the world
 *
 * Out:
 * nodes -- list of every node in grid ((dim.x+1)*(dim.y+1)*(dim.z+1))
 *
 */
__global__ void computeCellMassVelocityAndForceFast( Particle *particleData, Grid *grid, ParticleGridTempData *particleGridTempData, ParticleGrid::Node *nodes )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particleData[particleIdx];
    ParticleGridTempData &pgtd = particleGridTempData[particleIdx];

    glm::ivec3 currIJK;
    gridIndexToIJK(threadIdx.y, glm::ivec3(4,4,4), currIJK);
    currIJK.x += (int) pgtd.particleGridPos.x - 1; currIJK.y += (int) pgtd.particleGridPos.y - 1; currIJK.z += (int) pgtd.particleGridPos.z - 1;

    if (withinBoundsInclusive(currIJK, glm::ivec3(0,0,0), grid->dim)){
        ParticleGrid::Node &node = nodes[getGridIndex(currIJK, grid->dim+1)];

        float w;
        vec3 wg;
        vec3 nodePosition(currIJK.x, currIJK.y, currIJK.z);
        weightAndGradient(pgtd.particleGridPos-nodePosition, w, wg);

        atomicAdd(&node.mass, particle.mass*w);
        atomicAdd(&node.velocity, particle.velocity*particle.mass*w );
        atomicAdd(&node.force, pgtd.sigma*wg);
     }
}

/**
 * Called on each grid node.
 *
 * Updates the velocities of each grid node based on forces and collisions
 *
 * In:
 * nodes -- list of all nodes in the grid.
 * dt -- delta time, time step of simulation
 * colliders -- array of colliders in the scene.
 * numColliders -- number of colliders in the scene
 * worldParams -- Global parameters dealing with the physics of the world
 * grid -- parameters defining the grid
 *
 * Out:
 * nodes -- updated velocity and velocityChange
 *
 */
__global__ void updateNodeVelocities( ParticleGrid::Node *nodes, float dt, ImplicitCollider* colliders, int numColliders, MaterialConstants *material, Grid *grid, const vec3 gravity )
{
    int nodeIdx = blockIdx.x*blockDim.x + threadIdx.x;
    int gridI, gridJ, gridK;
    gridIndexToIJK(nodeIdx, gridI, gridJ, gridK, grid->dim+1);
    ParticleGrid::Node &node = nodes[nodeIdx];
    vec3 nodePosition = vec3(gridI, gridJ, gridK)*grid->h + grid->pos;

    float scale = ( node.mass > 1e-8 ) ? 1.f/node.mass : 0.f;

    node.velocity *= scale; //Have to normalize velocity by mass to conserve momentum

    // Update velocity with node force and gravity
    vec3 tmpVelocity = node.velocity + dt * ( node.force*scale + gravity );

    checkForAndHandleCollisions( colliders, numColliders, material->coeffFriction, nodePosition, tmpVelocity );
    node.velocityChange = tmpVelocity - node.velocity;
    node.velocity = tmpVelocity;
}

#define VEC2IVEC( V ) ( glm::ivec3((int)V.x, (int)V.y, (int)V.z) )

#define CLAMP( X, A, B ) ( (X < A) ? A : ((X > B) ? B : X) )

// Use weighting functions to compute particle velocity gradient and update particle velocity
__device__ void processGridVelocities( Particle &particle, Grid *grid, const ParticleGrid::Node *nodes, mat3 &velocityGradient, float alpha )
{
    const vec3 &pos = particle.position;
    const glm::ivec3 &dim = grid->dim;
    const float h = grid->h;

    // Compute neighborhood of particle in grid
    vec3 gridIndex = (pos - grid->pos) / h,
         gridMax = vec3::floor( gridIndex + vec3(2,2,2) ),
         gridMin = vec3::ceil( gridIndex - vec3(2,2,2) );
    glm::ivec3 maxIndex = glm::clamp( VEC2IVEC(gridMax), glm::ivec3(0,0,0), dim ),
               minIndex = glm::clamp( VEC2IVEC(gridMin), glm::ivec3(0,0,0), dim );

    // For computing particle velocity gradient:
    //      grad(v_p) = sum( v_i * transpose(grad(w_ip)) ) = [3x3 matrix]
    // For updating particle velocity:
    //      v_PIC = sum( v_i * w_ip )
    //      v_FLIP = v_p + sum( dv_i * w_ip )
    //      v = (1-alpha)*v_PIC _ alpha*v_FLIP
    vec3 v_PIC(0,0,0), dv_FLIP(0,0,0);
    int rowSize = dim.z+1;
    int pageSize = (dim.y+1)*rowSize;
    for ( int i = minIndex.x; i <= maxIndex.x; ++i ) {
        vec3 d, s;
        d.x = i - gridIndex.x;
        d.x *= ( s.x = ( d.x < 0 ) ? -1.f : 1.f );
        int pageOffset = i*pageSize;
        for ( int j = minIndex.y; j <= maxIndex.y; ++j ) {
            d.y = j - gridIndex.y;
            d.y *= ( s.y = ( d.y < 0 ) ? -1.f : 1.f );
            int rowOffset = pageOffset + j*rowSize;
            for ( int k = minIndex.z; k <= maxIndex.z; ++k ) {
                d.z = k - gridIndex.z;
                d.z *= ( s.z = ( d.z < 0 ) ? -1.f : 1.f );
                const ParticleGrid::Node &node = nodes[rowOffset+k];
                float w;
                vec3 wg;
                weightAndGradient( s, d, w, wg );
                velocityGradient += mat3::outerProduct( node.velocity, wg );
                // Particle velocities
                v_PIC += node.velocity * w;
                dv_FLIP += node.velocityChange * w;
            }
        }
    }
    particle.velocity = (1.f-alpha)*v_PIC + alpha*(particle.velocity+dv_FLIP);
}

__device__ void updateParticleDeformationGradients( Particle &particle, const mat3 &velocityGradient, float timeStep, MaterialConstants *mat )
{
    // Temporarily assign all deformation to elastic portion
    mat3 F = mat3::addIdentity( timeStep*velocityGradient ) * particle.elasticF;

    // Clamp the singular values
    mat3 W, S, Sinv, V;
    computeSVD( F, W, S, V );

    S = mat3( CLAMP( S[0], mat->criticalCompression, mat->criticalStretch ), 0.f, 0.f,
              0.f, CLAMP( S[4], mat->criticalCompression, mat->criticalStretch ), 0.f,
              0.f, 0.f, CLAMP( S[8], mat->criticalCompression, mat->criticalStretch ) );
    Sinv = mat3( 1.f/S[0], 0.f, 0.f,
                 0.f, 1.f/S[4], 0.f,
                 0.f, 0.f, 1.f/S[8] );

    // Compute final deformation components
    particle.elasticF = mat3::multiplyADBt( W, S, V );
    particle.plasticF = mat3::multiplyADBt( V, Sinv, W ) * particle.plasticF;
}

// NOTE: assumes particleCount % blockDim.x = 0, so tid is never out of range!
// criticalCompression = 1 - theta_c
// criticalStretch = 1 + theta_s
__global__ void updateParticlesFromGrid( Particle *particles, Grid *grid, const ParticleGrid::Node *nodes, float timeStep, ImplicitCollider *colliders, int numColliders, MaterialConstants *mat )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    Particle &particle = particles[tid];

    // Update particle velocities and fill in velocity gradient for deformation gradient computation
    mat3 velocityGradient = mat3( 0.f );
    processGridVelocities( particle, grid, nodes, velocityGradient, 0.95f );

    updateParticleDeformationGradients( particle, velocityGradient, timeStep, mat );

    checkForAndHandleCollisions( colliders, numColliders, mat->coeffFriction, particle.position, particle.velocity );

    particle.position += timeStep * (particle.velocity );

}


void updateParticles( const SimulationParameters &parameters,
                      Particle *particles, int numParticles,
                      Grid *grid, ParticleGrid::Node *nodes, int numNodes, ParticleGridTempData *particleGridTempData,
                      ImplicitCollider *colliders, int numColliders,
                      MaterialConstants *mat )
{

    static const int threadCount = 128;

    computeParticleGridTempData<<< numParticles / threadCount , threadCount >>>( particles, grid, mat, particleGridTempData );
    checkCudaErrors( cudaDeviceSynchronize() );

    // Clear grid data before update
    checkCudaErrors( cudaMemset(nodes, 0, numNodes*sizeof(ParticleGrid::Node)) );

    dim3 blockDim = dim3(numParticles / threadCount, 64);
    dim3 threadDim = dim3(threadCount/64, 64);
    computeCellMassVelocityAndForceFast<<< blockDim, threadDim >>>( particles, grid, particleGridTempData, nodes );
    checkCudaErrors( cudaDeviceSynchronize() );

    updateNodeVelocities<<< numNodes / threadCount, threadCount >>>( nodes, parameters.timeStep, colliders, numColliders, mat, grid, parameters.gravity );
    checkCudaErrors( cudaDeviceSynchronize() );

    updateParticlesFromGrid<<< numParticles / threadCount, threadCount >>>( particles, grid, nodes, parameters.timeStep, colliders, numColliders, mat );
    checkCudaErrors( cudaDeviceSynchronize() );

}
