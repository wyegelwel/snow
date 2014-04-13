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

#include "glm/common.hpp"
#include "glm/geometric.hpp"
#include "glm/gtc/random.hpp"

#define CUDA_INCLUDE
#include "cuda/functions.h"
#include "sim/particle.h"
#include "sim/grid.h"
#include "geometry/bbox.h"
#include "geometry/mesh.h"

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
    float det = glm::dot( e0, pVec );
    if ( fabsf(det) < 1e-8 ) return false;

    float invDet = 1.f / det;

    glm::vec3 tVec = rayO - v0;
    float u = glm::dot( tVec, pVec ) * invDet;
    if ( u < 0.f || u > 1.f ) return false;

    glm::vec3 qVec = glm::cross( tVec, e0 );
    float v = glm::dot( rayD, qVec ) * invDet;
    if ( v < 0.f || v > 1.f ) return false;

    t = glm::dot( e1, qVec ) * invDet;
    return t > 0.f;
}

__global__ void voxelizeMeshKernel( Tri *tris, int triCount, Grid grid, bool *flags )
{
    const glm::ivec3 &dim = grid.dim;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if ( x >= dim.x ) return;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ( y >= dim.y ) return;

    // Shoot ray in z-direction
    glm::vec3 origin = grid.pos + grid.h*glm::vec3(x+0.5f,y+0.5f,0.f);
    glm::vec3 direction = glm::vec3( 0.f, 0.f, 1.f );

    // Flag surface-intersecting voxels
    float t;
    int xyOffset = x*dim.y*dim.z + y*dim.z;
    for ( int i = 0; i < triCount; ++i ) {
        const Tri &tri = tris[i];
        if ( intersectTri(origin, direction, tri.v0, tri.v1, tri.v2, t) ) {
            int z = (int)(t/grid.h);
            flags[xyOffset+z] = true;
        }
    }

    // Scanline to fill inner voxels
    int end = xyOffset + dim.z;
    for ( int z = xyOffset; z < end; ++z ) {
        if ( flags[z] ) {
            do { z++; } while ( flags[z] && z < end );
            int zz = z;
            do { zz++; } while ( !flags[zz] && zz < end );
            if ( zz < end-1 ) {
                for ( int i = z; i < zz; ++i ) flags[i] = true;
                z = zz;
            } else break;
        }
    }

}

__global__ void sampleVoxelsKernel( curandState *states, unsigned int seed, bool *flags, int voxelCount, unsigned int *particleVoxels, int particleCount )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;

    // Rejection sample
    curand_init( seed, tid, 0, &states[tid] );
    unsigned int i;
    do {
        curandState localState = states[tid];
        i = curand(&localState) % voxelCount;
        states[tid] = localState;
    } while ( !flags[i] );

    particleVoxels[tid] = i;
}


__global__ void fillVoxelsKernel( curandState *states, Grid grid, unsigned int *particleVoxels, glm::vec3 *particlePositions, int particleCount )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;

    const glm::ivec3 &dim = grid.dim;

    // Convert voxel index to 3D index
    int i = (int)particleVoxels[tid];
    int x = i / (dim.y*dim.z);
    int y = (i-(x*dim.y*dim.z)) / (dim.z);
    int z = i - (y*dim.z) - (x*dim.y*dim.z);

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

    glm::vec3 min = grid.pos + grid.h*glm::vec3(x, y, z);
    glm::vec3 max = min + glm::vec3(grid.h,grid.h,grid.h);
    particlePositions[tid] = min + r*(max-min);
}

void fillMesh( cudaGraphicsResource **resource, int triCount, const Grid &grid, Particle *particles, int particleCount )
{
    // Get mesh data
    checkCudaErrors( cudaGraphicsMapResources(1, resource, 0) );
    Tri *tris;
    size_t size;
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void**)&tris, &size, *resource) );

    // Voxelize mesh
    dim3 blocks( (grid.dim.x+15)/16, (grid.dim.y+15)/16 );
    dim3 threads( 16, 16 );
    int voxelCount = grid.dim.x * grid.dim.y * grid.dim.z;
    bool *flags;
    checkCudaErrors( cudaMalloc((void**)&flags, voxelCount*sizeof(bool)) );
    checkCudaErrors( cudaMemset( (void*)flags, 0, voxelCount*sizeof(bool)) );
    voxelizeMeshKernel<<< blocks, threads >>>( tris, triCount, grid, flags );
    checkCudaErrors( cudaDeviceSynchronize() );
    checkCudaErrors( cudaGraphicsUnmapResources(1, resource, 0) );

    blocks.x = (particleCount+255)/256;
    blocks.y = 1;
    threads.x = 256;
    threads.y = 1;

    // Rejection sample voxels
    curandState *devStates;
    checkCudaErrors( cudaMalloc(&devStates, particleCount*sizeof(curandState)) );
    unsigned int *devParticleVoxels;
    checkCudaErrors( cudaMalloc(&devParticleVoxels, particleCount*sizeof(unsigned int)) );
    sampleVoxelsKernel<<< blocks, threads >>>( devStates, time(NULL), flags, voxelCount, devParticleVoxels, particleCount );
    checkCudaErrors( cudaDeviceSynchronize() );
    checkCudaErrors( cudaFree(flags) );

    // Randomly fill sampled voxels
    glm::vec3 *devPoints;
    checkCudaErrors( cudaMalloc((void**)&devPoints, particleCount*sizeof(glm::vec3)) );
    fillVoxelsKernel<<< blocks, threads >>>( devStates, grid, devParticleVoxels, devPoints, particleCount );
    checkCudaErrors( cudaDeviceSynchronize() );
    checkCudaErrors( cudaFree(devParticleVoxels) );
    checkCudaErrors( cudaFree(devStates) );

    // Copy back results
    glm::vec3 *hostPoints = new glm::vec3[particleCount];
    checkCudaErrors( cudaMemcpy(hostPoints, devPoints, particleCount*sizeof(glm::vec3), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaFree(devPoints) );
    for ( int i = 0; i < particleCount; ++i )
        particles[i].position = hostPoints[i];
    delete [] hostPoints;
}

#endif // MAX_CU
