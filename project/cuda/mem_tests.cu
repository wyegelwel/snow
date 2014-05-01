/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mem_tests.cu
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 1 May 2014
**
**************************************************************************/


#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "cuda/helpers.h"

#include "common/common.h"

#include "cuda/vector.h"
#include "cuda/matrix.h"

extern "C" { void testMemoryStuff(); }



struct Thing
{
    vec3 position;
    vec3 velocity;
    vec3 force;
    mat3 F;
    int id;
};

struct ThingList
{
    vec3 *positions;
    vec3 *velocities;
    vec3 *forces;
    mat3 *Fs;
    int *ids;
};

__global__ void initializeThings( Thing *things, int numThings )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numThings ) return;

    things[tid].position = vec3(0,0,0);
    things[tid].velocity = vec3(0,float(tid)/numThings,0);
    things[tid].force = vec3(0,-1,0);
    things[tid].F = mat3(1.f);
    things[tid].id = tid;
}

__global__ void processThings( Thing *things, int numThings )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numThings ) return;

    for ( int i = -10; i <= 10; ++i ) {
        int index = tid+i;
        if ( index >= 0 && index < numThings ) {
            things[tid].position += mat3::inverse(things[tid].F) * things[index].velocity;
        }
    }
}

__global__ void initializeThingList( ThingList *thingList, int numThings )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numThings ) return;

    thingList->positions[tid] = vec3(0,0,0);
    thingList->velocities[tid] = vec3(0,float(tid)/numThings,0);
    thingList->forces[tid] = vec3(0,-1,0);
    thingList->Fs[tid] = mat3(1.f);
    thingList->ids[tid] = tid;
}

__global__ void processThingList( ThingList *thingList, int numThings )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ( tid >= numThings ) return;

    for ( int i = -10; i <= 10; ++i ) {
        int index = tid+i;
        if ( index >= 0 && index < numThings ) {
            thingList->positions[tid] += mat3::inverse(thingList->Fs[tid]) * thingList->velocities[tid];
        }
    }
}

void testMemoryStuff()
{
    LOG( "TESTING MEMORY STUFF" );

    int N = 1000000;

    LOG( "%d blocks", (N+256-1)/256 );

    Thing *devThings;
    cudaMalloc((void**)&devThings, N*sizeof(Thing));

    ThingList *hostThingList = new ThingList;
    cudaMalloc( (void**)&hostThingList->positions, N*sizeof(vec3) );
    cudaMalloc( (void**)&hostThingList->velocities, N*sizeof(vec3) );
    cudaMalloc( (void**)&hostThingList->forces, N*sizeof(vec3) );
    cudaMalloc( (void**)&hostThingList->Fs, N*sizeof(mat3) );
    cudaMalloc( (void**)&hostThingList->ids, N*sizeof(int) );
    ThingList *devThingList;
    cudaMalloc( (void**)&devThingList, sizeof(ThingList) );
    cudaMemcpy( devThingList, hostThingList, sizeof(ThingList), cudaMemcpyHostToDevice );

    LOG( "Allocated %d things - %.2f MB", N, 2*N*sizeof(Thing)/1e6 );

    int threadCount = 256;
    dim3 blocks( (N+threadCount-1)/threadCount );
    dim3 threads( threadCount );

    TIME( "Testing Thing: ", "done.\n",
        LAUNCH( initializeThings<<<blocks,threads>>>(devThings,N) );
        LAUNCH( processThings<<<blocks,threads>>>(devThings,N) );
    );

    TIME( "Testing ThingList: ", "done.\n",
        LAUNCH( initializeThingList<<<blocks,threads>>>(devThingList,N) );
        LAUNCH( processThingList<<<blocks,threads>>>(devThingList,N) );
    );

    cudaFree( hostThingList->positions );
    cudaFree( hostThingList->velocities );
    cudaFree( hostThingList->forces );
    cudaFree( hostThingList->Fs );
    cudaFree( hostThingList->ids );
    delete hostThingList;
    cudaFree( devThingList );
    cudaFree( devThings );

    LOG( "DONE" );
}
