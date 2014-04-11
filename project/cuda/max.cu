/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   max.cu
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef MAX_CU
#define MAX_CU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/random.hpp>

#define CUDA_INCLUDE
#include "cuda/functions.h"
#include "sim/particle.h"
#include "geometry/bbox.h"
#include "geometry/mesh.h"

__device__ float dot( const glm::vec3 &a, const glm::vec3 &b )
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

/*
 * Moller, T, and Trumbore, B. Fast, Minimum Storage Ray/Triangle Intersection.
 */
__device__ bool intersectTri( const glm::vec3 &rayO, const glm::vec3 &rayD, 
                              const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2,
                              float &t )
{
    glm::vec3 e0 = v1 - v0;
    glm::vec3 e1 = v2 - v0;
    
    glm::vec3 pVec = glm::cross( rayD, e1 );
    float det = dot( e0, pVec );
    if ( fabsf(det) < 1e-8 ) return false;

    float invDet = 1.f / det;

    glm::vec3 tVec = rayO - v0;
    float u = dot(tVec, pVec) * invDet;
    if ( u < 0.f || u > 1.f ) return false;

    glm::vec3 qVec = glm::cross( tVec, e0 );
    float v = dot( rayD, qVec ) * invDet;
    if ( v < 0.f || v > 1.f ) return false;

    t = dot( e1, qVec ) * invDet;
    return true;
}

__global__ void voxelizeMeshKernel( Tri *tris, int triCount, const BBox box, const glm::ivec3 dim, float h, bool *flags )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if ( x >= dim.x ) return;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ( y >= dim.y ) return;

    // Shoot ray in z-direction
    glm::vec3 origin = box.min + glm::vec3( 0.5f + h*x, 0.5f + h*y, 0.f );
    glm::vec3 direction = glm::vec3( 0.f, 0.f, 1.f );

    // Flag surface-intersecting voxels
    float t;
    int xyOffset = x*dim.y*dim.z + y*dim.z;
    for ( int i = 0; i < triCount; ++i ) {
        const Tri &tri = tris[i];
        if ( intersectTri(origin, direction, tri.v0, tri.v1, tri.v2, t) ) {
            int z = (int)(t/h);
            flags[xyOffset+z] = !flags[xyOffset+z];
        }
    }

    // Scanline to fill inner voxels
    bool on = false;
    for ( int z = 0; z < dim.z; ++z ) {
        int i = xyOffset + z;
        on = ( flags[i] ) ? !on : on;
        flags[i] = ( on ) ? true : flags[i];
    }
}

__global__ void setupRNGKernel( curandState *states, unsigned long seed, int particleCount )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;
    curand_init( seed, tid, 0, &states[tid] );
}

__global__ void sampleVoxelsKernel( curandState *states, bool *flags, int voxelCount, unsigned int *particleVoxels, int particleCount )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;

    // Rejection sample
    unsigned int i;
    do {
        curandState localState = states[tid];
        i = curand(&localState) % voxelCount;
        states[tid] = localState;
    } while ( !flags[i] );

    particleVoxels[tid] = i;
}


__global__ void fillVoxelsKernel( curandState *states, const glm::ivec3 dim, float h, unsigned int *particleVoxels, glm::vec3 *particlePositions, int particleCount )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;

    // Convert voxel index to 3D index
    unsigned int i = particleVoxels[tid];
    unsigned int x = i / (dim.y*dim.z);
    unsigned int y = (i-x) / (dim.z);
    unsigned int z = i - y - x;

    // Generate 3 uniform floats
    glm::vec3 r;
    curandState localState = states[tid];
    r.x = curand_uniform(&localState);
    states[tid] = localState;
    localState = states[tid];
    r.y = curand_uniform(&localState);
    states[tid] = localState;
    localState = states[tid];
    r.z = curand_uniform(&localState);
    states[tid] = localState;

    glm::vec3 min = glm::vec3( 0.5f+x*h, 0, 0 );
    glm::vec3 max = min + glm::vec3( h, h, h );
    particlePositions[tid] = min;
}

void fillMesh( cudaGraphicsResource **resource, int triCount, const BBox &box, float h, Particle *particles, int particleCount )
{
    // Get mesh data
    checkCudaErrors( cudaGraphicsMapResources(1, resource, 0) );
    Tri *tris;
    size_t size;
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void**)&tris, &size, *resource) );

    // Compute grid dimensions
    glm::vec3 dimf = glm::round( (box.max-box.min)/h );
    glm::ivec3 dim = glm::ivec3( (int)dimf.x, (int)dimf.y, (int)dimf.z );
    int voxelCount = dim.x * dim.y * dim.z;
    printf( "%d x %d x %d = %d voxels\n", dim.x, dim.y, dim.z, voxelCount ); fflush(stdout);

    dim3 blocks(1,1,1), threads(1,1,1);

    // Voxelize mesh

    blocks.x = (dim.x+15)/16;
    blocks.y = (dim.y+15)/16;
    threads.x = 16;
    threads.y = 16;
    bool *flags;
    checkCudaErrors( cudaMalloc((void**)&flags, voxelCount*sizeof(bool)) );
    checkCudaErrors( cudaMemset( (void*)flags, 0, voxelCount*sizeof(bool)) );
    voxelizeMeshKernel<<< blocks, threads >>>( tris, triCount, box, dim, h, flags );
    checkCudaErrors( cudaGraphicsUnmapResources(1, resource, 0) );

    // Randomly fill object voxels

    blocks.x = (particleCount+511)/512;
    blocks.y = 1;
    threads.x = 512;
    threads.y = 1;

    // Seed random integers
    printf( "Seeding random numbers...\n" ); fflush(stdout);
    curandState *devStates;
    checkCudaErrors( cudaMalloc(&devStates, particleCount*sizeof(curandState)) );
    setupRNGKernel<<< blocks, threads >>>( devStates, time(NULL), particleCount );

    // Rejection sample voxels
    printf( "Sampling voxels...\n" ); fflush(stdout);
    unsigned int *devParticleVoxels;
    checkCudaErrors( cudaMalloc(&devParticleVoxels, particleCount*sizeof(unsigned int)) );
    sampleVoxelsKernel<<< blocks, threads >>>( devStates, flags, voxelCount, devParticleVoxels, particleCount );
    checkCudaErrors( cudaFree(flags) );

    printf( "Filling voxels...\n" ); fflush(stdout);
    glm::vec3 *devPoints;
    checkCudaErrors( cudaMalloc((void**)&devPoints, particleCount*sizeof(glm::vec3)) );
    fillVoxelsKernel<<< blocks, threads >>>( devStates, dim, h, devParticleVoxels, devPoints, particleCount );
    checkCudaErrors( cudaFree(devParticleVoxels) );
    checkCudaErrors( cudaFree(devStates) );

    // Copy back results
    printf( "Copying results...\n" ); fflush(stdout);
    glm::vec3 *hostPoints = new glm::vec3[particleCount];
    checkCudaErrors( cudaMemcpy(hostPoints, devPoints, particleCount*sizeof(glm::vec3), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaFree(devPoints) );
    for ( int i = 0; i < particleCount; ++i ) {
        particles[i].position = hostPoints[i];
        printf( "%f %f %f\n", hostPoints[i].x, hostPoints[i].y, hostPoints[i].z ); fflush(stdout);
    }
    delete [] hostPoints;
}

#endif // MAX_CU
