#ifndef WIL_CU
#define WIL_CU

#define GLM_FORCE_CUDA
#include <stdio.h>
#include "common/common.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher
#endif

extern "C"  {
    void testComputedF();
}

#include "sim/particle.h"
#include "sim/particlegridnode.h"
#include "sim/particlegrid.h"
#include "sim/caches.h"
#include "cuda/vector.h";
#include "cuda/weighting.h"
#include "cuda/decomposition.h"


//#include "cuda/implicit.h"

__host__ void cudaMallocAndCopy(void *cudaDst, void *src, int size){
    checkCudaErrors(cudaMalloc((void **) &cudaDst, size));
    checkCudaErrors(cudaMemcpy(cudaDst, src, size, cudaMemcpyHostToDevice));
}

__host__ void testComputedF(){
    const static int numParticles = 1;
    Particle particles[numParticles];
    particles[0].mass = 1e-7; particles[0].position = vec3(.2f);

    Grid grid;
    grid.dim = glm::ivec3(2,2,2); grid.h = .2f; grid.pos = vec3(0.f);

    Node *nodes = new Node[grid.nodeCount()];
    for (int i = 0; i < grid.nodeCount(); i++){
        nodes[i].velocity = vec3(1.f);
    }


    ParticleCache particleCache[numParticles];
    vec3 du[grid.nodeCount()];
    du[0] = vec3(1.f);


    Particle *devParticles;
    Grid *devGrid;
    Node *devNodes;
    ParticleCache *devParticlesCache;
    vec3 *devDu;

    cudaMallocAndCopy(devParticles, particles,  numParticles*sizeof(Particle));

    cudaMallocAndCopy(devGrid, &grid, sizeof(Grid));

    cudaMallocAndCopy(devNodes, nodes, grid.nodeCount()*sizeof(Node));

    cudaMallocAndCopy(devParticlesCache, particleCache, numParticles*sizeof(ParticleCache));

    cudaMallocAndCopy(devDu, du, grid.nodeCount()*sizeof(vec3));


    computedF<<<numParticles,1 >>>(devParticles, devGrid, 1, devNodes, devDu, devParticlesCache);

    delete[] du;
    delete[] nodes;
    checkCudaErrors(cudaFree(devParticles));
    checkCudaErrors(cudaFree(devGrid));
    checkCudaErrors(cudaFree(devNodes));
    checkCudaErrors(cudaFree(devParticlesCache));
    checkCudaErrors(cudaFree(devDu));

}




#endif // WIL_CU
