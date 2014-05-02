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

#include "cuda/helpers.h"
#include "cuda/vector.h"

#define CUDA_INCLUDE
#include "common/common.h"
#include "common/math.h"
#include "cuda/functions.h"
#include "sim/particle.h"
#include "geometry/grid.h"
#include "geometry/bbox.h"
#include "cuda/noise.h"
#include "cuda/snowtypes.h"

#include "glm/gtc/random.hpp"

struct Tri {
    vec3 v0, n0;
    vec3 v1, n1;
    vec3 v2, n2;
};


/*
 * Moller, T, and Trumbore, B. Fast, Minimum Storage Ray/Triangle Intersection.
 */
__device__ int intersectTri(const vec3 &v1, const vec3 &v2, const vec3 &v3,
                            const vec3 &O, const vec3 &D, float &t)
{
    vec3 e1, e2;  //Edge1, Edge2
    vec3 P, Q, T;
    float det, inv_det, u, v;
    e1 = v2-v1;
    e2 = v3-v1;
    P = vec3::cross(D,e2);
    det = vec3::dot(e1,P);
    if(det > -1e-8 && det < 1e-8) return 0;
    inv_det = 1.f / det;
    T = O-v1;
    u = vec3::dot(T, P) * inv_det;
    if(u < 0.f || u > 1.f) return 0;
    Q = vec3::cross(T, e1);
    v = vec3::dot(D,Q)*inv_det;
    if(v < 0.f || u + v  > 1.f) return 0;
    t = vec3::dot(e2, Q) * inv_det;
    if(t > 1e-8) { //ray intersection
        return 1;
    }
    // No hit, no win
    return 0;
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
        if ( intersectTri(tri.v0, tri.v1, tri.v2, origin, direction, t) ) {
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
    particle.velocity = vec3(0,-1,0);
    particle.material = Material();
    particles[tid] = particle;
}

void fillMesh( cudaGraphicsResource **resource, int triCount, const Grid &grid, Particle *particles, int particleCount, float targetDensity, int materialPreset)
{
    // Get mesh data
    cudaGraphicsMapResources( 1, resource, 0 );
    Tri *devTris;
    size_t size;
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void**)&devTris, &size, *resource) );

    // Voxelize mesh
    int x = grid.dim.x > 16 ? MAX( 1, MIN(16, grid.dim.x/8)) : 1;
    int y = grid.dim.y > 16 ? MAX( 1, MIN(16, grid.dim.y/8)) : 1;
    dim3 blocks( (grid.dim.x+x-1)/x, (grid.dim.y+y-1)/y ), threads( x, y );
    int voxelCount = grid.dim.x * grid.dim.y * grid.dim.z;
    bool *devFlags;
    checkCudaErrors( cudaMalloc((void**)&devFlags, voxelCount*sizeof(bool)) );
    checkCudaErrors( cudaMemset((void*)devFlags, 0, voxelCount*sizeof(bool)) );
    voxelizeMeshKernel<<< blocks, threads >>>( devTris, triCount, grid, devFlags );
    checkCudaErrors( cudaDeviceSynchronize() );

    int powerOfTwo = (int)(log2f(voxelCount)+1);
    int reductionSize = 1 << powerOfTwo;
    int *devReduction;
    checkCudaErrors( cudaMalloc((void**)&devReduction, reductionSize*sizeof(int)) );
    initReduction<<< (reductionSize+511)/512, 512 >>>( devFlags, voxelCount, devReduction, reductionSize );
    checkCudaErrors( cudaDeviceSynchronize() );
    for ( int i = 0; i < powerOfTwo-1; ++i ) {
        int size = 1 << (powerOfTwo-i-1);
        reduce<<< (size+511)/512, 512 >>>( devReduction, size );
        checkCudaErrors( cudaDeviceSynchronize() );
    }
    int count;
    checkCudaErrors( cudaMemcpy(&count, devReduction, sizeof(int), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaFree(devReduction) );
    float volume = count*grid.h*grid.h*grid.h;
    float particleMass = targetDensity * volume / particleCount;
    LOG( "Average %.2f particles per grid cell.", float(particleCount)/count );
    LOG( "Target Density: %.1f kg/m3 -> Particle Mass: %g kg", targetDensity, particleMass );


    // Randomly fill mesh voxels and copy back resulting particles
    curandState *devStates;
    checkCudaErrors( cudaMalloc(&devStates, particleCount*sizeof(curandState)) );
    Particle *devParticles;
    checkCudaErrors( cudaMalloc((void**)&devParticles, particleCount*sizeof(Particle)) );
    fillMeshVoxelsKernel<<< (particleCount+511)/512, 512 >>>( devStates, time(NULL), grid, devFlags, devParticles, particleMass, particleCount );
    checkCudaErrors( cudaDeviceSynchronize() );

    switch (materialPreset)
    {
    case 0:
        break;
    case 1:
        LAUNCH( applyChunky<<<(particleCount+511)/512, 512>>>(devParticles,particleCount) ); // TODO - we could use the uisettings materialstiffness here
        LOG( "Chunky applied" );
        break;
    default:
        break;
    }

    checkCudaErrors( cudaMemcpy(particles, devParticles, particleCount*sizeof(Particle), cudaMemcpyDeviceToHost) );

    checkCudaErrors( cudaFree(devFlags) );
    checkCudaErrors( cudaFree(devStates) );
    checkCudaErrors( cudaGraphicsUnmapResources(1, resource, 0) );
}


#if 0 // mesh filling algorithm #2

__device__ bool isInMesh(vec3 pos, int seed, Tri *tris, int triCount)
{
    // returns true if point lies inside mesh.
    // this is a simple scanline check, returns true if number of intersections
    // between point and top of boudning box is odd.
    // one unfortunate case is handling glancing intersections,
    // so in that case we want to sample 2 random rays
    //vec3 d1(0,1,0);
    vec3 d1(0.110432, 0.993884, 0.);
    vec3 d2(0,-1,0);
    int c = 1; // number of intersections. start with one so the parity check will work when no intersections happen.
    float t;
    for ( int i = 0; i < triCount; ++i ) {
        const Tri &tri = tris[i];
        int test1 =intersectTri(tri.v0, tri.v1, tri.v2, pos, d1,t);
        int test2 = intersectTri(tri.v0, tri.v1, tri.v2, pos, d2, t);
        c += int(test1 && test2);
       //c += test1;
    }
    return (c+1)%2;
}



__global__ void fillMeshKernel2( Tri *tris, int triCount, vec3 bbox_min, vec3 bbox_size, Particle *particles, float particleMass, int particleCount)
{
    // naive mesh filling algorithm
    // this approach results in less successes than scanline projection from Z plane and sorting the intersections,
    // but implementation is much simpler and all particles should be filled within a couple iterations.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= particleCount ) return;

    vec3 pos;
    int ii = 20; // halton incrementer. discard first 20
    int rejected = 0;
    bool accept = false;
    while (!accept) {
        ii += 1; // rejected, keep going
        rejected += 1;
        vec3 u = vec3(halton(tid+ii, 3), halton(tid+ii, 5), halton(tid+ii, 7));
        pos = bbox_min + u * bbox_size; // sample random point within bounding box
        accept = isInMesh(pos, tid+ii, tris, triCount);
    }
    printf("%d\n", rejected);
    Particle particle;
    particle.mass = particleMass;
    particle.position = pos;
    particles[tid] = particle;
}

/**
 * alternative mesh filling scheme - rejection sampling the bounding box.
 * works really well if mesh occupies majority of bounding box but really slow otherwise.
 * CUDA ends up timing out
 */
void fillMesh2( cudaGraphicsResource **resource, int triCount, const Grid &grid, Particle *particles, int particleCount, float targetDensity)
{
    // Get mesh data    
    cudaGraphicsMapResources( 1, resource, 0 );
    Tri *devTris;
    size_t size;
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void**)&devTris, &size, *resource) );

    Particle *devParticles;
    checkCudaErrors( cudaMalloc((void**)&devParticles, particleCount*sizeof(Particle)) );
    float volume = particleCount*grid.h*grid.h*grid.h;
    float particleMass = targetDensity * volume / particleCount;
    BBox box(grid);

    fillMeshKernel2<<< (particleCount+511)/512, 512 >>>( devTris, triCount, box.min(), box.size(), devParticles, particleMass, particleCount );

    checkCudaErrors( cudaDeviceSynchronize() );
    checkCudaErrors( cudaMemcpy(particles, devParticles, particleCount*sizeof(Particle), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaGraphicsUnmapResources(1, resource, 0) );
}

#endif


#endif // MESH_CU
