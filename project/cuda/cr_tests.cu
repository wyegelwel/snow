/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   simulation.cu
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 30 Apr 2014
**
**************************************************************************/

#include "common/common.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "math.h"

#define CUDA_INCLUDE
#include "cuda/helpers.h"
#include "cuda/vector.h"
#include "cuda/matrix.h"
#include "sim/caches.h"

extern "C" { void testConjugateResidual(); }

static const vec3 b[4] = {
vec3( 0.9429602449, 0.2418600860, 0.9989322688 ),
vec3( 0.5826938151, 0.1832790006, 0.3868454219 ),
vec3( 0.1896735289, 0.4107706730, 0.5946800689 ),
vec3( 0.7165860931, 0.4868914824, 0.3095898178 )
};

static const vec3 x[4] = {
vec3( 43.8136982011, -17.9855783836, 19.6472561118 ),
vec3( -1.0808269295, -0.0794553643, 39.2803330659 ),
vec3( -42.6068133318, 3.8876065036, 7.2348756408 ),
vec3( 19.3079428688, -6.1138815252, -49.8075338592 )
};

static const mat3 E[16] = {
mat3( 3.6010201538, 3.2481459017, 1.7735144119, 3.2481459017, 4.3333424188, 2.3574551475, 1.7735144119, 2.3574551475, 2.5411289239 ),
mat3( 2.9823245541, 3.4558550097, 1.7876522064, 2.4542385213, 2.5099955809, 1.8711027441, 2.0664784871, 2.5885569417, 2.0247212986 ),
mat3( 2.5628214352, 2.8546450180, 2.4592391986, 2.7135185873, 2.8471693020, 1.8854611322, 3.0727753562, 4.1035350948, 2.9332252629 ),
mat3( 2.0698187809, 2.6492924433, 2.1657740276, 1.7825022559, 1.9186826331, 1.6043852262, 3.2858897309, 3.3478418432, 2.3582968466 ),
mat3( 2.9823245541, 2.4542385213, 2.0664784871, 3.4558550097, 2.5099955809, 2.5885569417, 1.7876522064, 1.8711027441, 2.0247212986 ),
mat3( 4.3742311911, 2.8960870745, 3.1582376355, 2.8960870745, 3.6170889108, 2.8516008429, 3.1582376355, 2.8516008429, 3.5020055310 ),
mat3( 3.0558608031, 3.6296419197, 3.1602657003, 3.2228142351, 3.0843475640, 2.6885500923, 3.6180189884, 3.3176900406, 3.4919048581 ),
mat3( 3.2901763250, 2.2216196874, 3.0896229489, 2.3335517070, 2.3048689222, 2.5239333816, 3.6121754592, 2.3632590181, 3.2642955077 ),
mat3( 2.5628214352, 2.7135185873, 3.0727753562, 2.8546450180, 2.8471693020, 4.1035350948, 2.4592391986, 1.8854611322, 2.9332252629 ),
mat3( 3.0558608031, 3.2228142351, 3.6180189884, 3.6296419197, 3.0843475640, 3.3176900406, 3.1602657003, 2.6885500923, 3.4919048581 ),
mat3( 4.6416857558, 3.2260400212, 4.0184071645, 3.2260400212, 3.1196579369, 3.4115038484, 4.0184071645, 3.4115038484, 5.2115995495 ),
mat3( 2.8728408272, 2.5114012072, 3.7567292400, 2.5538966178, 2.0373771307, 2.7959987444, 2.2751206154, 2.8426556555, 3.7353007151 ),
mat3( 2.0698187809, 1.7825022559, 3.2858897309, 2.6492924433, 1.9186826331, 3.3478418432, 2.1657740276, 1.6043852262, 2.3582968466 ),
mat3( 3.2901763250, 2.3335517070, 3.6121754592, 2.2216196874, 2.3048689222, 2.3632590181, 3.0896229489, 2.5239333816, 3.2642955077 ),
mat3( 2.8728408272, 2.5538966178, 2.2751206154, 2.5114012072, 2.0373771307, 2.8426556555, 3.7567292400, 2.7959987444, 3.7353007151 ),
mat3( 4.0168094858, 2.2950560759, 3.6252732452, 2.2950560759, 2.2574134685, 2.4274841604, 3.6252732452, 2.4274841604, 5.0234253381 )
};

__global__ void printKernel( NodeCache *caches, int numNodes, NodeCache::Offset offset )
{
    for ( int i = 0; i < numNodes; ++i ) {
        const NodeCache &cache = caches[i];
        const vec3 &v = cache[offset];
        printf( "        %.10f %.10f %.10f\n", v.x, v.y, v.z );
    }
}

__global__ void _initializeVKernel( NodeCache *caches )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    caches[tid].v = vec3( 0, 0, 0 );
}

__global__ void _initializeRPKernel( NodeCache *caches, vec3 *devB )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    caches[tid].r = devB[tid] - caches[tid].r;
    caches[tid].p = caches[tid].r;
}

__global__ void _initializeApKernel( NodeCache *caches )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    caches[tid].Ap = caches[tid].Ar;
}

__global__ void _updateVRKernel( NodeCache *caches, float alpha )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    caches[tid].v += alpha * caches[tid].p;
    caches[tid].r -= alpha * caches[tid].Ap;
}

__global__ void _updatePApResidualKernel( NodeCache *caches, float beta )
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    caches[tid].p = caches[tid].r + beta * caches[tid].p;
    caches[tid].Ap = caches[tid].Ar + beta * caches[tid].Ap;
    caches[tid].scratch = vec3::dot( caches[tid].r, caches[tid].r );
}

__global__ void _innerProductKernel( NodeCache *caches, int numNodes, NodeCache::Offset uOffset, NodeCache::Offset vOffset )
{
    float sum = 0.f;
    for ( int i = 0; i < numNodes; ++i ) {
        const NodeCache &cache = caches[i];
        sum += vec3::dot( cache[uOffset], cache[vOffset] );
    }
    caches[0].scratch = sum;
}

__host__ float _innerProduct( NodeCache *caches, int numNodes, NodeCache::Offset uOffset, NodeCache::Offset vOffset )
{
    _innerProductKernel<<<1,1>>>( caches, numNodes, uOffset, vOffset ); cudaDeviceSynchronize();
    float result;
    cudaMemcpy( &result, &(caches[0].scratch), sizeof(float), cudaMemcpyDeviceToHost );
    return result;
}

__global__ void _scratchSumKernel( NodeCache *caches, int numNodes )
{
    float sum = 0.f;
    for ( int i = 0; i < numNodes; ++i ) {
        sum += caches[i].scratch;
    }
    caches[0].scratch = sum;
}

__host__ float _scratchSum( NodeCache *caches, int numNodes )
{
    LAUNCH( _scratchSumKernel<<<1,1>>>(caches,numNodes) );
    float result;
    cudaMemcpy( &result, &(caches[0].scratch), sizeof(float), cudaMemcpyDeviceToHost );
    return result;
}

__global__ void computeEuKernel( NodeCache *caches, int numNodes, const mat3 *devE, NodeCache::Offset uOffset, NodeCache::Offset resultOffset )
{
    for ( int i = 0; i < numNodes; ++i ) {
        NodeCache &resultCache = caches[i];
        vec3 &result = resultCache[resultOffset];
        result = vec3( 0, 0, 0 );
        for ( int j = 0; j < numNodes; ++j ) {
            NodeCache &uCache = caches[j];
            vec3 &u = uCache[uOffset];
            const mat3 &m = devE[4*i+j];
            result += m * u;
        }
    }
}

void testConjugateResidual()
{
    int numNodes = 4;

    NodeCache *devCaches;
    checkCudaErrors( cudaMalloc((void**)&devCaches, numNodes*sizeof(NodeCache)) );
    checkCudaErrors( cudaMemset(devCaches, 0, numNodes*sizeof(NodeCache)) );

    vec3 *devB;
    checkCudaErrors( cudaMalloc((void**)&devB, numNodes*sizeof(vec3)) );
    checkCudaErrors( cudaMemcpy(devB, (vec3*)(b), numNodes*sizeof(vec3), cudaMemcpyHostToDevice) );

    mat3 *devE;
    checkCudaErrors( cudaMalloc((void**)&devE, numNodes*numNodes*sizeof(mat3)) );
    checkCudaErrors( cudaMemcpy(devE, (mat3*)(E), numNodes*numNodes*sizeof(mat3), cudaMemcpyHostToDevice) );

    _initializeVKernel<<<numNodes,1>>>( devCaches ); cudaDeviceSynchronize();
    computeEuKernel<<<1,1>>>( devCaches, numNodes, devE, NodeCache::V, NodeCache::R ); cudaDeviceSynchronize();
    _initializeRPKernel<<<numNodes,1>>>( devCaches, devB ); cudaDeviceSynchronize();
    computeEuKernel<<<1,1>>>( devCaches, numNodes, devE, NodeCache::R, NodeCache::AR ); cudaDeviceSynchronize();
    _initializeApKernel<<<numNodes,1>>>( devCaches ); cudaDeviceSynchronize();

    int k = 0;
    float residual;
    do {

        float alphaNum = _innerProduct( devCaches, numNodes, NodeCache::R, NodeCache::AR );
        float alphaDen = _innerProduct( devCaches, numNodes, NodeCache::AP, NodeCache::AP );
        float alpha = ( fabsf(alphaDen) > 0.f ) ? alphaNum / alphaDen : 0.f;

        float betaDen = alphaNum;
        _updateVRKernel<<<numNodes,1>>>( devCaches, alpha ); cudaDeviceSynchronize();
        computeEuKernel<<<1,1>>>( devCaches, numNodes, devE, NodeCache::R, NodeCache::AR ); cudaDeviceSynchronize();
        float betaNum = _innerProduct( devCaches, numNodes, NodeCache::R, NodeCache::AR );
        float beta = ( fabsf(betaDen) > 0.f ) ? betaNum / betaDen : 0.f;

        _updatePApResidualKernel<<<numNodes,1>>>( devCaches, beta ); cudaDeviceSynchronize();
        residual = _scratchSum( devCaches, numNodes );

        LOG( "k = %3d, alpha = %10.7f, beta = %10.7f, residual = %g", k, alpha, beta, residual );

    } while ( k++ < 100 && residual > 1e-12 );

    NodeCache *caches = new NodeCache[numNodes];
    checkCudaErrors( cudaMemcpy(caches, devCaches, numNodes*sizeof(NodeCache), cudaMemcpyDeviceToHost) );

    bool passed = residual < 1e-8;

    printf( "\nExpected:\n" );
    for ( int i = 0; i < numNodes; ++i ) {
        const vec3 &v = x[i];
        printf( "        %.10f %.10f %.10f\n", v.x, v.y, v.z );
    }
    printf( "\nGot:\n" );
    LAUNCH( printKernel<<<1,1>>>(devCaches, numNodes, NodeCache::V) );
    printf( "\n" );

    if ( passed ) {
        LOG( "CONJUGATE RESIDUAL: PASSED (%d iterations)", k );
    } else {
        LOG( "CONJUGATE RESIDUAL: FAILED" );
    }

    cudaFree( devCaches );
    cudaFree( devB );
    cudaFree( devE );
}
