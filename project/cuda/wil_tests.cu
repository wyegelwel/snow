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

__device__ __host__ void printMat3( const mat3 &mat ) {
    // prints by rows
    for (int j=0; j<3; ++j) // g3d stores column-major
    {
        for (int i=0; i<3; ++i)
        {
            printf("%f   ", mat[3*i+j]);
        }
        printf("\n");
    }
    printf("\n");
}

#define cudaMallocAndCopy( dst, src, size )                    \
({                                                             \
    cudaMalloc((void**) &dst, size);                           \
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);        \
})

#define TEST( toCheck, msg, failExprs)      \
({                                          \
    if (toCheck){                           \
        printf("[PASSED]: %s\n", msg);      \
    }else{                                  \
        printf("[FAILED]: %s\n", msg);      \
        failExprs;                          \
    }                                       \
})


__host__ void simpleTestComputedF(){
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
    for (int i = 0; i < grid.nodeCount(); i++){
        du[i] = vec3(1.f);
    }



    Particle *devParticles;
    Grid *devGrid;
    Node *devNodes;
    ParticleCache *devParticlesCache;
    vec3 *devDu;

    cudaMallocAndCopy(devParticles, particles, numParticles*sizeof(Particle));

    cudaMallocAndCopy(devGrid, &grid, sizeof(Grid));

    cudaMallocAndCopy(devNodes, nodes, grid.nodeCount()*sizeof(Node));

    cudaMallocAndCopy(devParticlesCache, particleCache, numParticles*sizeof(ParticleCache));

    cudaMallocAndCopy(devDu, du, grid.nodeCount()*sizeof(vec3));

    computedF<<<numParticles,1 >>>(devParticles, devGrid, 1, devNodes, devDu, devParticlesCache);
    checkCudaErrors(cudaDeviceSynchronize());



    checkCudaErrors(cudaMemcpy(particleCache, devParticlesCache, numParticles*sizeof(ParticleCache), cudaMemcpyDeviceToHost));


    TEST(mat3::equals(mat3(0.0f), particleCache[0].dF), "Simple computedF test", printMat3(particleCache[0].dF););


//    delete[] du;
    delete[] nodes;
    checkCudaErrors(cudaFree(devParticles));
    checkCudaErrors(cudaFree(devGrid));
    checkCudaErrors(cudaFree(devNodes));
    checkCudaErrors(cudaFree(devParticlesCache));
    checkCudaErrors(cudaFree(devDu));
}

__host__ void testComputedF(){
    printf("Testing computedF\n");

    simpleTestComputedF();

    Grid grid;
    int dim = 15;
    grid.dim = glm::ivec3(dim,dim,dim); grid.h = .2f; grid.pos = vec3(0.f);

    const static int numParticles = 256;
    Particle particles[numParticles];
    for (int i = 0; i < numParticles; i++){
        particles[i].mass = 1e-7; particles[i].position = vec3((float) i/(dim+1) * grid.h);
    }


    Node *nodes = new Node[grid.nodeCount()];
    for (int i = 0; i < grid.nodeCount(); i++){
        nodes[i].velocity = vec3(1.f);
    }

    ParticleCache particleCache[numParticles];
    vec3 *du = new vec3[grid.nodeCount()];
    for (int i = 0; i < grid.nodeCount(); i++){
        du[i] = vec3(1.f);
    }

    Particle *devParticles;
    Grid *devGrid;
    Node *devNodes;
    ParticleCache *devParticlesCache;
    vec3 *devDu;

    cudaMallocAndCopy(devParticles, particles, numParticles*sizeof(Particle));

    cudaMallocAndCopy(devGrid, &grid, sizeof(Grid));

    cudaMallocAndCopy(devNodes, nodes, grid.nodeCount()*sizeof(Node));

    cudaMallocAndCopy(devParticlesCache, particleCache, numParticles*sizeof(ParticleCache));

    cudaMallocAndCopy(devDu, du, grid.nodeCount()*sizeof(vec3));

    computedF<<<numParticles,1 >>>(devParticles, devGrid, 1, devNodes, devDu, devParticlesCache);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(particleCache, devParticlesCache, numParticles*sizeof(ParticleCache), cudaMemcpyDeviceToHost));

    mat3 dF0 = mat3(0.347222f,0.347222f,0.347222f, 0.347222f,0.347222f,0.347222f, 0.347222f,0.347222f,0.347222f);
    mat3 dF20 = mat3(0.0f);
    mat3 dF249 = mat3(-0.153244f,-0.153244f,-0.153244f, -0.153244f,-0.153244f,-0.153244f, -0.153244f,-0.153244f,-0.153244f);

    TEST(mat3::equals(dF0, particleCache[0].dF), "Complex computedF test 1", printMat3(particleCache[0].dF););
    TEST(mat3::equals(dF20, particleCache[20].dF), "Complex computedF test 2", printMat3(particleCache[20].dF););
    TEST(mat3::equals(dF249, particleCache[249].dF), "Complex computedF test 3", printMat3(particleCache[249].dF););


    delete[] du;
    delete[] nodes;
    checkCudaErrors(cudaFree(devParticles));
    checkCudaErrors(cudaFree(devGrid));
    checkCudaErrors(cudaFree(devNodes));
    checkCudaErrors(cudaFree(devParticlesCache));
    checkCudaErrors(cudaFree(devDu));

}


__host__ void simpleTestComputedF(){
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
    for (int i = 0; i < grid.nodeCount(); i++){
        du[i] = vec3(1.f);
    }



    Particle *devParticles;
    Grid *devGrid;
    Node *devNodes;
    ParticleCache *devParticlesCache;
    vec3 *devDu;

    cudaMallocAndCopy(devParticles, particles, numParticles*sizeof(Particle));

    cudaMallocAndCopy(devGrid, &grid, sizeof(Grid));

    cudaMallocAndCopy(devNodes, nodes, grid.nodeCount()*sizeof(Node));

    cudaMallocAndCopy(devParticlesCache, particleCache, numParticles*sizeof(ParticleCache));

    cudaMallocAndCopy(devDu, du, grid.nodeCount()*sizeof(vec3));

    computedF<<<numParticles,1 >>>(devParticles, devGrid, 1, devNodes, devDu, devParticlesCache);
    checkCudaErrors(cudaDeviceSynchronize());



    checkCudaErrors(cudaMemcpy(particleCache, devParticlesCache, numParticles*sizeof(ParticleCache), cudaMemcpyDeviceToHost));


    TEST(mat3::equals(mat3(0.0f), particleCache[0].dF), "Simple computedF test", printMat3(particleCache[0].dF););


//    delete[] du;
    delete[] nodes;
    checkCudaErrors(cudaFree(devParticles));
    checkCudaErrors(cudaFree(devGrid));
    checkCudaErrors(cudaFree(devNodes));
    checkCudaErrors(cudaFree(devParticlesCache));
    checkCudaErrors(cudaFree(devDu));
}

__host__ void testComputedF(){
    printf("Testing computedF\n");

    simpleTestComputedF();

    Grid grid;
    int dim = 15;
    grid.dim = glm::ivec3(dim,dim,dim); grid.h = .2f; grid.pos = vec3(0.f);

    const static int numParticles = 256;
    Particle particles[numParticles];
    for (int i = 0; i < numParticles; i++){
        particles[i].mass = 1e-7; particles[i].position = vec3((float) i/(dim+1) * grid.h);
    }


    Node *nodes = new Node[grid.nodeCount()];
    for (int i = 0; i < grid.nodeCount(); i++){
        nodes[i].velocity = vec3(1.f);
    }

    ParticleCache particleCache[numParticles];
    vec3 *du = new vec3[grid.nodeCount()];
    for (int i = 0; i < grid.nodeCount(); i++){
        du[i] = vec3(1.f);
    }

    Particle *devParticles;
    Grid *devGrid;
    Node *devNodes;
    ParticleCache *devParticlesCache;
    vec3 *devDu;

    cudaMallocAndCopy(devParticles, particles, numParticles*sizeof(Particle));

    cudaMallocAndCopy(devGrid, &grid, sizeof(Grid));

    cudaMallocAndCopy(devNodes, nodes, grid.nodeCount()*sizeof(Node));

    cudaMallocAndCopy(devParticlesCache, particleCache, numParticles*sizeof(ParticleCache));

    cudaMallocAndCopy(devDu, du, grid.nodeCount()*sizeof(vec3));

    computedF<<<numParticles,1 >>>(devParticles, devGrid, 1, devNodes, devDu, devParticlesCache);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(particleCache, devParticlesCache, numParticles*sizeof(ParticleCache), cudaMemcpyDeviceToHost));

    mat3 dF0 = mat3(0.347222f,0.347222f,0.347222f, 0.347222f,0.347222f,0.347222f, 0.347222f,0.347222f,0.347222f);
    mat3 dF20 = mat3(0.0f);
    mat3 dF249 = mat3(-0.153244f,-0.153244f,-0.153244f, -0.153244f,-0.153244f,-0.153244f, -0.153244f,-0.153244f,-0.153244f);

    TEST(mat3::equals(dF0, particleCache[0].dF), "Complex computedF dF test 1", printMat3(particleCache[0].dF););
    TEST(mat3::equals(dF20, particleCache[20].dF), "Complex computedF dF test 2", printMat3(particleCache[20].dF););
    TEST(mat3::equals(dF249, particleCache[249].dF), "Complex computedF dF test 3", printMat3(particleCache[249].dF););

    mat3 FeHat0 = mat3(1.347222f,0.347222f,0.347222f, 0.347222f,1.347222f,0.347222f, 0.347222f,0.347222f,1.347222f);
    mat3 FeHat93 = mat3(1.0f);
    TEST(mat3::equals(FeHat0, particleCache[0].FeHat), "Complex computedF FeHat test 1", printMat3(particleCache[0].FeHat););
    TEST(mat3::equals(FeHat93, particleCache[93].FeHat), "Complex computedF FeHat test 2", printMat3(particleCache[93].FeHat););

    mat3 ReHat0 = mat3(1.0f);
    TEST(mat3::equals(ReHat0, particleCache[0].ReHat), "Complex computedF Re test 2", printMat3(particleCache[0].ReHat););
    mat3 SeHat0 = mat3(1.347222f,0.347222f,0.347222f, 0.347222f,1.347222f,0.347222f, 0.347222f,0.347222f,1.347222f);
    TEST(mat3::equals(SeHat0, particleCache[0].SeHat), "Complex computedF Se test 2", printMat3(particleCache[0].SeHat););

    delete[] du;
    delete[] nodes;
    checkCudaErrors(cudaFree(devParticles));
    checkCudaErrors(cudaFree(devGrid));
    checkCudaErrors(cudaFree(devNodes));
    checkCudaErrors(cudaFree(devParticlesCache));
    checkCudaErrors(cudaFree(devDu));

}


__global__ void testComputedRKernel(){
    mat3 dF(0.0f);
    mat3 Se(1.0);
    mat3 Re(1.0);
    mat3 dR;
    mat3 expecteddR = mat3(0,0,0);
    computedR(dF, Se, Re, dR);
    TEST(mat3::equals(dR, expecteddR), "Simple computeR test", printMat3(dR););


    dF = mat3(1,1,1, 1,1,1, 1,1,1);
    Se = mat3(1.0,-.1,3,  -.1,1.2,0,  3,0,.3);
    Re = mat3(-1.0f,0,0,   0,-1,0,   0,0,1);
    expecteddR = mat3( 0.000000000000000,  1.130247578040904,   1.474703982777180,
                       -1.130247578040904,   0.000000000000000,  -0.828848223896663,
                       1.474703982777180,  -0.828848223896663,                   0);
    computedR(dF, Se, Re, dR);
    TEST(mat3::equals(dR, expecteddR), "More complex computeR test", printMat3(dR););
}

__host__ void testComputedR(){
    testComputedRKernel<<<1,1>>>();
    cudaDeviceSynchronize();
}


__global__ void testcompute_dJF_invTransKernel(){
    mat3 F(0.0);
    mat3 dF(0.0f);
    mat3 dF_inv(0);
    mat3 expecteddF_inv = mat3(0);
    compute_dJF_invTrans(F, dF, dF_inv);
    TEST(mat3::equals(dF_inv, expecteddF_inv), "Simple compute_dJF_invTrans test", printMat3(dF_inv););




    F = mat3( 7,    -2,     0,
             -2,     6,    -2,
              0,    -2,     5);
    dF = mat3(17.0000,   -2.9000,      0.0f,
              -2.9000,   -1.0000,   -1.0000,
                 0.0f,   -1.0000,    7.0000);
    expecteddF_inv = mat3(33.0000, 28.5000, 7.8000, 28.5000, 134.0000, 41.0000, 7.8000, 41.0000, 83.4000);
    compute_dJF_invTrans(F, dF, dF_inv);
    TEST(mat3::equals(dF_inv, expecteddF_inv), "Complex compute_dJF_invTrans test", printMat3(dF_inv););
}

__host__ void testcompute_dJF_invTrans(){
    testcompute_dJF_invTransKernel<<<1,1>>>();
    cudaDeviceSynchronize();
}


#endif // WIL_CU
