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

#include "cuda/vector.cu"

#define CUDA_INCLUDE
#include "common/common.h"
#include "common/math.h"
#include "cuda/functions.h"
#include "sim/particle.h"
#include "geometry/grid.h"

struct Tri {
    vec3 v0, n0;
    vec3 v1, n1;
    vec3 v2, n2;
};

/*
 * Moller, T, and Trumbore, B. Fast, Minimum Storage Ray/Triangle Intersection.
 */
__device__ bool intersectTri( const vec3 &rayO, const vec3 &rayD,
                              const vec3 &v0, const vec3 &v1, const vec3 &v2,
                              float &t )
{
    vec3 e0 = v1 - v0;
    vec3 e1 = v2 - v0;

    vec3 pVec = vec3::cross( rayD, e1 );
    float det = vec3::dot( e0, pVec );
    if ( fabsf(det) < 1e-8 ) return false;

    float invDet = 1.f / det;

    vec3 tVec = rayO - v0;
    float u = vec3::dot( tVec, pVec ) * invDet;
    if ( u < 0.f || u > 1.f ) return false;

    vec3 qVec = vec3::cross( tVec, e0 );
    float v = vec3::dot( rayD, qVec ) * invDet;
    if ( v < 0.f || v > 1.f ) return false;

    t = vec3::dot( e1, qVec ) * invDet;
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
    vec3 origin = grid.pos + grid.h * vec3( x+0.5f, y+0.5f, 0.f );
    vec3 direction = vec3( 0.f, 0.f, 1.f );

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

__global__ void initReduction( bool *flags, int voxelCount, int *reduction, int reductionSize )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= reductionSize ) return;
    reduction[tid] = ( tid < voxelCount ) ? flags[tid] : 0;
}

__global__ void reduce( int *reduction, int size )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= size ) return;
    reduction[tid] += reduction[tid+size];
}

__global__ void fillMeshVoxelsKernel( curandState *states, unsigned int seed, Grid grid, bool *flags, Particle *particles, float particleMass, int particleCount )
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
    vec3 r = vec3( curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState) );
    vec3 min = grid.pos + grid.h * vec3( x, y, z );
    vec3 max = min + vec3( grid.h, grid.h, grid.h );

    Particle particle;
    particle.mass = particleMass;
    particle.position = min + r*(max-min);
    particles[tid] = particle;
}

void fillMesh( cudaGraphicsResource **resource, int triCount, const Grid &grid, Particle *particles, int particleCount, float targetDensity )
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

    int powerOfTwo = (int)(log2f(voxelCount)+1);
    int reductionSize = 1 << powerOfTwo;
    int *devReduction;
    cudaMalloc( (void**)&devReduction, reductionSize*sizeof(int) );
    initReduction<<< (reductionSize+511)/512, 512 >>>( devFlags, voxelCount, devReduction, reductionSize );
    cudaDeviceSynchronize();
    for ( int i = 0; i < powerOfTwo-1; ++i ) {
        int size = 1 << (powerOfTwo-i-1);
        reduce<<< (size+511)/512, 512 >>>( devReduction, size );
        cudaDeviceSynchronize();
    }
    int count;
    cudaMemcpy( &count, devReduction, sizeof(int), cudaMemcpyDeviceToHost );
    cudaFree( devReduction );
    float volume = count*grid.h*grid.h*grid.h;
    float particleMass = targetDensity * volume / particleCount;
    LOG( "Average %.2f particles per grid cell.", float(particleCount)/count );
    LOG( "Target Density: %.1f kg/m3 -> Particle Mass: %g kg", targetDensity, particleMass );

    // Randomly fill mesh voxels and copy back resulting particles
    curandState *devStates;
    cudaMalloc( &devStates, particleCount*sizeof(curandState) );
    Particle *devParticles;
    cudaMalloc( (void**)&devParticles, particleCount*sizeof(Particle) );
    fillMeshVoxelsKernel<<< (particleCount+511)/512, 512 >>>( devStates, time(NULL), grid, devFlags, devParticles, particleMass, particleCount );
    checkCudaErrors( cudaDeviceSynchronize() );
    cudaMemcpy( particles, devParticles, particleCount*sizeof(Particle), cudaMemcpyDeviceToHost );

    cudaFree( devFlags );
    cudaFree( devStates );
    cudaGraphicsUnmapResources( 1, resource, 0 );

}

#endif // MESH_CU
