/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mesh.cu
**   Author: mliberma
**   Created: 13 Apr 2014
**
**************************************************************************/

#ifndef MESH_CU
#define MESH_CU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "glm/common.hpp"
#include "glm/geometric.hpp"

#define CUDA_INCLUDE
#include "common/common.h"
#include "common/math.h"
#include "cuda/functions.h"
#include "geometry/mesh.h"
#include "sim/particle.h"
#include "sim/grid.h"

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
    glm::vec3 origin = grid.pos + grid.h * glm::vec3( x+0.5f, y+0.5f, 0.f );
    glm::vec3 direction = glm::vec3( 0.f, 0.f, 1.f );

    // Flag surface-intersecting voxels
    float t;
    int xyOffset = x*dim.y*dim.z + y*dim.z, z;
    for ( int i = 0; i < triCount; ++i ) {
        const Tri &tri = tris[i];
        if ( intersectTri(origin, direction, tri.v0, tri.v1, tri.v2, t) ) {
            z = (int)(t/grid.h);
            flags[xyOffset+z] = true;
        }
    }

    // Scanline to fill inner voxels
    int end = xyOffset + dim.z, zz;
    for ( int z = xyOffset; z < end; ++z ) {
        if ( flags[z] ) {
            do { z++; } while ( flags[z] && z < end );
            zz = z;
            do { zz++; } while ( !flags[zz] && zz < end );
            if ( zz < end - 1 ) {
                for ( int i = z; i < zz; ++i ) flags[i] = true;
                z = zz;
            } else break;
        }
    }

}

__global__ void fillMeshVoxelsKernel( curandState *states, unsigned int seed, Grid grid, bool *flags, Particle *particles, int particleCount )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;

    curandState &localState = states[tid];
    curand_init( seed, tid, 0, &localState );

    const glm::ivec3 &dim = grid.dim;

    // Rejection sample
    unsigned int i;
    unsigned int voxelCount = dim.x * dim.y * dim.z;
    do { i = curand(&localState) % voxelCount; } while ( !flags[i] );

    // Get 3D voxel index
    unsigned int x = i / (dim.y*dim.z);
    unsigned int y = (i - x*dim.y*dim.z) / dim.z;
    unsigned int z = i - y*dim.z - x*dim.y*dim.z;

    // Generate random point in voxel cube
    glm::vec3 r = glm::vec3( curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState) );
    glm::vec3 min = grid.pos + grid.h * glm::vec3( x, y, z );
    glm::vec3 max = min + glm::vec3( grid.h, grid.h, grid.h );
    particles[tid].position = min + r*(max-min);
}

void fillMesh( cudaGraphicsResource **resource, int triCount, const Grid &grid, Particle *particles, int particleCount )
{
    // Get mesh data
    cudaGraphicsMapResources( 1, resource, 0 );
    Tri *devTris;
    size_t size;
    cudaGraphicsResourceGetMappedPointer( (void**)&devTris, &size, *resource );

    // Voxelize mesh
    int x = grid.dim.x > 16 ? MAX( 1, MIN(16, grid.dim.x/8)) : 1;
    int y = grid.dim.y > 16 ? MAX( 1, MIN(16, grid.dim.y/8)) : 1;
    dim3 blocks( (grid.dim.x+x-1)/x, (grid.dim.y+y-1)/y ), threads( x, y );
    int voxelCount = grid.dim.x * grid.dim.y * grid.dim.z;
    bool *devFlags;
    cudaMalloc( (void**)&devFlags, voxelCount*sizeof(bool) );
    cudaMemset( (void*)devFlags, 0, voxelCount*sizeof(bool) );
    voxelizeMeshKernel<<< blocks, threads >>>( devTris, triCount, grid, devFlags );
    checkCudaErrors( cudaDeviceSynchronize() );

    // Randomly fill mesh voxels and copy back resulting particles
    curandState *devStates;
    cudaMalloc( &devStates, particleCount*sizeof(curandState) );
    Particle *devParticles;
    cudaMalloc( (void**)&devParticles, particleCount*sizeof(Particle) );
    fillMeshVoxelsKernel<<< (particleCount+511)/512, 512 >>>( devStates, time(NULL), grid, devFlags, devParticles, particleCount );
    checkCudaErrors( cudaDeviceSynchronize() );
    cudaMemcpy( particles, devParticles, particleCount*sizeof(Particle), cudaMemcpyDeviceToHost );

    cudaFree( devFlags );
    cudaFree( devStates );
    cudaGraphicsUnmapResources( 1, resource, 0 );

}

#endif // MESH_CU
