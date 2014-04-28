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

#include "math.h"

#define CUDA_INCLUDE
#include "sim/collider.h"
#include "sim/material.h"
#include "sim/parameters.h"
#include "sim/particle.h"
#include "sim/particlegridnode.h"

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


// Chain to compute the volume of the particle

/**
 * Part of one time operation to compute particle volumes. First rasterize particle masses to grid
 *
 * Operation done over Particles over grid node particle affects
 */
__global__ void computeCellMasses( Particle *particleData, Grid *grid, float* cellMasses )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particleData[particleIdx];

    glm::ivec3 currIJK;
    gridIndexToIJK( threadIdx.y, glm::ivec3(4,4,4), currIJK );
    vec3 particleGridPos = (particle.position - grid->pos) / grid->h;
    currIJK.x += (int) particleGridPos.x - 1; currIJK.y += (int) particleGridPos.y - 1; currIJK.z += (int) particleGridPos.z - 1;

    if ( withinBoundsInclusive(currIJK, glm::ivec3(0,0,0), grid->dim) ) {
        vec3 nodePosition( currIJK.x, currIJK.y, currIJK.z );
        vec3 dx = vec3::abs( particleGridPos - nodePosition );
        float w = weight( dx );
        atomicAdd( &cellMasses[getGridIndex(currIJK, grid->dim+1)], particle.mass*w );
     }
}

/**
 * Computes the particle's density * grid's volume. This needs to be separate from computeCellMasses(...) because
 * we need to wait for ALL threads to sync before computing the density
 *
 * Operation done over Particles over grid node particle affects
 */
__global__ void computeParticleDensity( Particle *particleData, Grid *grid, float *cellMasses )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    Particle &particle = particleData[particleIdx];

    glm::ivec3 currIJK;
    gridIndexToIJK( threadIdx.y, glm::ivec3(4,4,4), currIJK );
    vec3 particleGridPos = ( particle.position - grid->pos ) / grid->h;
    currIJK.x += (int) particleGridPos.x - 1; currIJK.y += (int) particleGridPos.y - 1; currIJK.z += (int) particleGridPos.z - 1;

    if ( withinBoundsInclusive(currIJK, glm::ivec3(0,0,0), grid->dim) ) {
        vec3 nodePosition( currIJK.x, currIJK.y, currIJK.z );
        vec3 dx = vec3::abs( particleGridPos - nodePosition );
        float w = weight( dx );
        float gridVolume = grid->h * grid->h * grid->h;
        atomicAdd( &particle.volume, cellMasses[getGridIndex(currIJK, grid->dim+1)] * w / gridVolume ); //fill volume with particle density. Then in final step, compute volume
     }
}

/**
 * Computes the particle's volume. Assumes computeParticleDensity(...) has just been called.
 *
 * Operation done over particles
 */
__global__ void computeParticleVolume( Particle *particleData )
{
    int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    Particle &particle = particleData[particleIdx];
    particle.volume = particle.mass / particle.volume; // Note: particle.volume is assumed to be the (particle's density ) before we compute it correctly
}

__host__ void initializeParticleVolumes( Particle *particles, int numParticles, Grid *grid, int numNodes )
{
    float *devCellMasses;
    checkCudaErrors( cudaMalloc( (void**)&devCellMasses, numNodes*sizeof(float) ) );
    cudaMemset( devCellMasses, 0, numNodes*sizeof(float) );

    static const int threadCount = 128;

    dim3 blockDim = dim3( numParticles / threadCount, 64 );
    dim3 threadDim = dim3( threadCount/64, 64 );

    computeCellMasses<<< blockDim, threadDim >>>( particles, grid, devCellMasses );
    checkCudaErrors( cudaDeviceSynchronize() );

    computeParticleDensity<<< blockDim, threadDim >>>( particles, grid, devCellMasses );
    checkCudaErrors( cudaDeviceSynchronize() );

    computeParticleVolume<<< numParticles / threadCount, threadCount >>>( particles );
    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaFree( devCellMasses) );
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
    sigma = (2*muFp*mat3::multiplyABt(Fe-Re, Fe) + mat3(lambdaFp*(Jep-1)*Jep)) * -particle.volume;
}


__global__ void computeParticleGridTempData ( Particle *particleData, Grid *grid, MaterialConstants *material, ParticleTempData *particleGridTempData )
{
    int particleIdx = blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particleData[particleIdx];
    ParticleTempData &pgtd = particleGridTempData[particleIdx];

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

__device__ __forceinline__
void atomicAdd( mat3 *add, mat3 toAdd )
{
    atomicAdd(&(add->data[0]), toAdd[0]);
    atomicAdd(&(add->data[1]), toAdd[1]);
    atomicAdd(&(add->data[2]), toAdd[2]);
    atomicAdd(&(add->data[3]), toAdd[3]);
    atomicAdd(&(add->data[4]), toAdd[4]);
    atomicAdd(&(add->data[5]), toAdd[5]);
    atomicAdd(&(add->data[6]), toAdd[6]);
    atomicAdd(&(add->data[7]), toAdd[7]);
    atomicAdd(&(add->data[8]), toAdd[8]);
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
__global__ void computeCellMassVelocityAndForceFast( Particle *particleData, Grid *grid, ParticleTempData *particleGridTempData, ParticleGridNode *nodes )
{
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particleData[particleIdx];
    ParticleTempData &pgtd = particleGridTempData[particleIdx];

    glm::ivec3 currIJK;
    gridIndexToIJK(threadIdx.y, glm::ivec3(4,4,4), currIJK);
    currIJK.x += (int) pgtd.particleGridPos.x - 1; currIJK.y += (int) pgtd.particleGridPos.y - 1; currIJK.z += (int) pgtd.particleGridPos.z - 1;

    if (withinBoundsInclusive(currIJK, glm::ivec3(0,0,0), grid->dim)){
        ParticleGridNode &node = nodes[getGridIndex(currIJK, grid->dim+1)];

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
__global__ void updateNodeVelocities( ParticleGridNode *nodes, float dt, ImplicitCollider* colliders, int numColliders, MaterialConstants *material, Grid *grid )
{
    int nodeIdx = blockIdx.x*blockDim.x + threadIdx.x;
    ParticleGridNode &node = nodes[nodeIdx];

    if (node.mass > 1e-12){
        float scale = 1.f/node.mass;

        node.velocity *= scale; //Have to normalize velocity by mass to conserve momentum

        // Update velocity with node force
        vec3 tmpVelocity = node.velocity + dt*node.force*scale;

        // Handle collisions
        int gridI, gridJ, gridK;
        gridIndexToIJK(nodeIdx, gridI, gridJ, gridK, grid->dim+1);
        vec3 nodePosition = vec3(gridI, gridJ, gridK)*grid->h + grid->pos;
        checkForAndHandleCollisions( colliders, numColliders, material->coeffFriction, nodePosition, tmpVelocity );

        node.velocityChange = tmpVelocity - node.velocity;
        node.velocity = tmpVelocity;
    }
}

#define VEC2IVEC( V ) ( glm::ivec3((int)V.x, (int)V.y, (int)V.z) )

// Use weighting functions to compute particle velocity gradient and update particle velocity
__device__ void processGridVelocities( Particle &particle, Grid *grid, const ParticleGridNode *nodes, mat3 &velocityGradient, float alpha )
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
                const ParticleGridNode &node = nodes[rowOffset+k];
                float w;
                vec3 wg;
                weightAndGradient( -s, d, w, wg );
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
    particle.elasticF = mat3::addIdentity( timeStep*velocityGradient ) * particle.elasticF;

    // Clamp the singular values
    mat3 W, S, Sinv, V;
    computeSVD( particle.elasticF, W, S, V );

    // FAST COMPUTATION:

    S = mat3( CLAMP( S[0], mat->criticalCompression, mat->criticalStretch ), 0.f, 0.f,
              0.f, CLAMP( S[4], mat->criticalCompression, mat->criticalStretch ), 0.f,
              0.f, 0.f, CLAMP( S[8], mat->criticalCompression, mat->criticalStretch ) );

    Sinv = mat3( 1.f/S[0], 0.f, 0.f,
                 0.f, 1.f/S[4], 0.f,
                 0.f, 0.f, 1.f/S[8] );

    // Compute final deformation components
    particle.plasticF = mat3::multiplyADBt( V, Sinv, W ) * particle.elasticF * particle.plasticF;
    particle.elasticF = mat3::multiplyADBt( W, S, V );

//     // MORE ACCURATE COMPUTATION:

//    S[0] = CLAMP( S[0], mat->criticalCompression, mat->criticalStretch );
//    S[4] = CLAMP( S[4], mat->criticalCompression, mat->criticalStretch );
//    S[8] = CLAMP( S[8], mat->criticalCompression, mat->criticalStretch );

//    particle.elasticF = W * S * mat3::transpose( V );
//    particle.plasticF = V * mat3::inverse( S ) * mat3::transpose( W ) * particle.elasticF * particle.plasticF;

}

// NOTE: assumes particleCount % blockDim.x = 0, so tid is never out of range!
// criticalCompression = 1 - theta_c
// criticalStretch = 1 + theta_s
__global__ void updateParticlesFromGrid( Particle *particles, Grid *grid, const ParticleGridNode *nodes, float timeStep, ImplicitCollider *colliders, int numColliders, MaterialConstants *mat, vec3 gravity )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    Particle &particle = particles[tid];

    // Update particle velocities and fill in velocity gradient for deformation gradient computation
    mat3 velocityGradient = mat3( 0.f );
    processGridVelocities( particle, grid, nodes, velocityGradient, 0.95f );

    updateParticleDeformationGradients( particle, velocityGradient, timeStep, mat );

    // Do this before collision test!
    particle.velocity += timeStep * gravity;

    checkForAndHandleCollisions( colliders, numColliders, mat->coeffFriction, particle.position, particle.velocity );

    particle.position += timeStep * ( particle.velocity );
}


/**
 * Approximate the shading normal of each particle
 * for each particle,find its corresponding grid node, then
 * approximate X,Y,Z component of mass gradient by examining average between the
 * two nearest neighbors on that axis (6 total - left,right,up,down,front,back)
 * then normalize the vector. If particle l2 norm is < epsilon (i.e. gradient is zero in all directions)
 * then just pick a random direction instead of normalizing.
 *
 * Particles that lie on boundary of surface will hopefully
 * have continuous normal.
 *
 * we could loop over grid nodes to cache the gradients at each node?
 * to reduce aliasing we could perturn the normals slightly. after all, snow is slightly scattery...
 */
__global__ void updateParticleNormals(Particle *particles, Grid *grid, const ParticleGridNode *nodes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Particle &particle = particles[tid];
    const vec3 &pos = particle.position;
    const glm::ivec3 &dim = grid->dim;
    const float h = grid->h;

    vec3 gridIndex = (pos - grid->pos) / h,
         gridMax = vec3::floor( gridIndex + vec3(1,1,1) ),
         gridMin = vec3::ceil( gridIndex - vec3(1,1,1) );
    glm::ivec3 maxIndex = glm::clamp( VEC2IVEC(gridMax), glm::ivec3(0,0,0), dim ),
               minIndex = glm::clamp( VEC2IVEC(gridMin), glm::ivec3(0,0,0), dim );

    //glm::ivec3 ijk = VEC2IVEC((pos - grid->pos) / h);


    // +x,-x,+y,-y,+z,-z axis-aligned components of negative gradient within grid
    // for higher resolution we could average over a larger neighborhood
    vec3 n = vec3(0,1,0);
//    ParticleGridNode &node;
//    int i1, i2; // grid indices of neighboring components
//    float c1,c2;
//    for (int a=0;a<3;a++)
//    {
//        glm::ivec3 da(0);
//        da[a]=-1;

//        da[1]=+1;
//        glm::ivec3 neighbor = ijk + glm::ivec3();

//        n[i] = (c1+c2)*.5;
//    }


//    //


    particle.normal = n;
}

/**
 * Called over particles over nodes the particle affects. (numParticles * 64)
 *
 * Recommended:
 *  dim3 blockDim = dim3(numParticles / threadCount, 64);
 *  dim3 threadDim = dim3(threadCount/64, 64);
 *
 **/
__global__ void computedF(Particle *particles, Grid *grid, vec3 *du, mat3 *dFs){
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particles[particleIdx];
    mat3 &dF = dFs[particleIdx];

    vec3 particleGridPos = (particle.position - grid->pos)/grid->h;
    glm::ivec3 currIJK;
    gridIndexToIJK(threadIdx.y, glm::ivec3(4,4,4), currIJK);
    currIJK.x += (int) particleGridPos.x - 1; currIJK.y += (int) particleGridPos.y - 1; currIJK.z += (int) particleGridPos.z - 1;

    if (withinBoundsInclusive(currIJK, glm::ivec3(0,0,0), grid->dim)){
        vec3 du_j = du[getGridIndex(currIJK, grid->dim+1)];

        float w;
        vec3 wg;
        vec3 nodePosition(currIJK.x, currIJK.y, currIJK.z);
        weightAndGradient(particleGridPos-nodePosition, w, wg);

        atomicAdd(&dF, mat3::outerProduct(du_j, wg) * particle.elasticF);
     }

}

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

// We will want to cache Re and Se since we will use it many times per time step
__global__ void computeAp(Particle *particles, Grid *grid, vec3 *du, mat3 *dFs, mat3 *Aps){
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particles[particleIdx];
    mat3 &dF = dFs[particleIdx];
    mat3 &Ap = Aps[particleIdx];

    mat3 &Fp = particle.plasticF; //for the sake of making the code look like the math
    mat3 &Fe = particle.elasticF;

    float Jpp = mat3::determinant(Fp);
    float Jep = mat3::determinant(Fe);

    mat3 Re, Se;
    computePD(Fe, Re, Se);

    float muFp = material->mu*__expf(material->xi*(1-Jpp));
    float lambdaFp = material->lambda*__expf(material->xi*(1-Jpp));

    mat3 dRe = Re; // Need to actually compute dRe

    mat3 jFe_invTrans = Jep*mat3::transpose(mat3::inverse(Fe));


    Ap = (2*muFp*(dF - dRe) +lambdaFp*jFe_invTrans*mat3::innerProduct(jFe_invTrans, dF) + lambdaFp*(Jep - 1));

//    sigma = (2*muFp*(Fe-Re)*mat3::transpose(Fe)+lambdaFp*(Jep-1)*Jep*mat3(1.0f)) * (particle.volume);
//    sigma = (2*muFp*mat3::multiplyABt(Fe-Re, Fe) + mat3(lambdaFp*(Jep-1)*Jep)) * -particle.volume;
}

void updateParticles( const SimulationParameters &parameters,
                      Particle *particles, int numParticles,
                      Grid *grid, ParticleGridNode *nodes, int numNodes, ParticleTempData *particleGridTempData,
                      ImplicitCollider *colliders, int numColliders,
                      MaterialConstants *mat,
                      bool doShading)
{
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    static const int threadCount = 128;

    computeParticleGridTempData<<< numParticles / threadCount , threadCount >>>( particles, grid, mat, particleGridTempData );
    checkCudaErrors( cudaDeviceSynchronize() );

    // Clear grid data before update
    checkCudaErrors( cudaMemset(nodes, 0, numNodes*sizeof(ParticleGridNode)) );

    dim3 blockDim = dim3(numParticles / threadCount, 64);
    dim3 threadDim = dim3(threadCount/64, 64);
    computeCellMassVelocityAndForceFast<<< blockDim, threadDim >>>( particles, grid, particleGridTempData, nodes );
    checkCudaErrors( cudaDeviceSynchronize() );

    if (doShading)
    {
        updateParticleNormals<<< numParticles/threadCount, threadCount >>> (particles, grid, nodes);
        checkCudaErrors( cudaDeviceSynchronize() );
    }

    updateNodeVelocities<<< numNodes / threadCount, threadCount >>>( nodes, parameters.timeStep, colliders, numColliders, mat, grid );
    checkCudaErrors( cudaDeviceSynchronize() );

    updateParticlesFromGrid<<< numParticles / threadCount, threadCount >>>( particles, grid, nodes, parameters.timeStep, colliders, numColliders, mat, parameters.gravity );
    checkCudaErrors( cudaDeviceSynchronize() );
}
