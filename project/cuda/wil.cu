/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   wil.cu
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef WIL_CU
#define WIL_CU

#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h> // prevents syntax errors on __global__ and __device__, among other things
#include "common/common.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher
#endif

extern "C"  {
    void testColliding();
    void testColliderNormal();
    void testGridMath();
    void timingTests();

}

// Grid math

#include "tim.cu" // should really be snow.cu or grid.cu depending on how we break it up
#include "decomposition.cu"
#include "weighting.cu"
#include "sim/particlegridnode.h"
#include "sim/material.h"
#include "cuda/collider.h"
#include "sim/particle.h"
//#include "matrix.cu"
//#include "vector.cu"

;


__host__ __device__ void computeSigma(Particle &particle, MaterialConstants *mat, mat3 &sigma){
    mat3 &Fp = particle.plasticF; //for the sake of making the code look like the math
    mat3 &Fe = particle.elasticF;

    float Jpp = mat3::determinant(Fp);
    float Jep = mat3::determinant(Fe);

    mat3 Re;
    computePD(Fe, Re);

    float muFp = mat->mu*exp(mat->xi*(1-Jpp));
    float lambdaFp = mat->lambda*exp(mat->xi*(1-Jpp));

    sigma = (2*muFp*(Fe-Re)*mat3::transpose(Fe)+lambdaFp*(Jep-1)*Jep*mat3(1.0f)) * (particle.volume);
}


__device__ void atomicAdd(vec3 *add, vec3 toAdd){
    atomicAdd(&(add->x), toAdd.x);
    atomicAdd(&(add->y), toAdd.y);
    atomicAdd(&(add->z), toAdd.z);
}

/**
 * Called on each particle.
 *
 * Each particle adds it's mass, velocity and force contribution to the grid nodes within 2h of itself.
 *
 * In:
 * particleData -- list of particles
 * grid -- Stores grid paramters
 * MaterialConstants -- Global parameters dealing with the physics of the world
 *
 * Out:
 * nodes -- list of every node in grid ((dim.x+1)*(dim.y+1)*(dim.z+1))
 *
 */
__global__ void computeCellMassVelocityAndForce(Particle *particleData, Grid *grid, MaterialConstants *mat, ParticleGridNode *nodes){
    int particleIdx = blockIdx.x*blockDim.x + threadIdx.x;
    Particle &particle = particleData[particleIdx];

    mat3 sigma;
    computeSigma(particle, mat, sigma);

    vec3 particleGridPos = (particle.position - grid->pos)/grid->h;
    glm::ivec3 min = glm::ivec3(std::ceil(particleGridPos.x - 2), std::ceil(particleGridPos.y - 2), std::ceil(particleGridPos.z - 2));
    glm::ivec3 max = glm::ivec3(std::floor(particleGridPos.x + 2), std::floor(particleGridPos.y + 2), std::floor(particleGridPos.z + 2));


    // Apply particles contribution of mass, velocity and force to surrounding nodes
    min = glm::max(glm::ivec3(0.0f), min);
    max = glm::min(grid->dim, max);
    for (int i = min.x; i <= max.x; i++){
        for (int j = min.y; j <= max.y; j++){
            for (int k = min.z; k <= max.z; k++){
                int currIdx = getGridIndex(i, j, k, grid->dim+1);
                ParticleGridNode &node = nodes[currIdx];

                float w;
                vec3 wg;
                weightAndGradient(particleGridPos - vec3(i, j, k), w, wg);

                atomicAdd(&node.mass, particle.mass*w);
                atomicAdd(&node.velocity, particle.velocity*particle.mass*w);
                atomicAdd(&node.force, sigma*wg);
            }
        }
    }
}


__host__ __device__ __forceinline__
bool withinBoundsInclusive( const float &v, const float &min, const float &max ) { return (v >= min && v <= max); }

__host__ __device__ __forceinline__
bool withinBoundsInclusive( const glm::ivec3 &v, const glm::ivec3 &min, const glm::ivec3 &max ) { return  withinBoundsInclusive(v.x, min.x, max.x)
                                                                                                            && withinBoundsInclusive(v.y, min.y, max.y)
                                                                                                            && withinBoundsInclusive(v.z, min.z, max.z);}
__global__ void computeParticleGridTempData(Particle *particleData, Grid *grid, MaterialConstants *mat, ParticleTempData *particleGridTempData){
    int particleIdx = blockIdx.x*blockDim.x + threadIdx.x;
    Particle &particle = particleData[particleIdx];
    ParticleTempData &pgtd = particleGridTempData[particleIdx];


//    vec3 particleGridPos = (particle.position - grid->pos)/grid->h;
//    pgtd.gridIJK = glm::ivec3((int) particleGridPos.x, (int) particleGridPos.y, (int) particleGridPos.z);
    pgtd.particleGridPos = (particle.position - grid->pos)/grid->h;
    computeSigma(particle, mat, pgtd.sigma);
}

/**
 * Called on each particle.
 *
 * Each particle adds it's mass, velocity and force contribution to the grid nodes within 2h of itself.
 *
 * In:
 * particleData -- list of particles
 * grid -- Stores grid paramters
 * MaterialConstants -- Global parameters dealing with the physics of the world
 *
 * Out:
 * nodes -- list of every node in grid ((dim.x+1)*(dim.y+1)*(dim.z+1))
 *
 */
__global__ void computeCellMassVelocityAndForceFast(Particle *particleData, Grid *grid, ParticleTempData *particleGridTempData, ParticleGridNode *nodes){
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
        atomicAdd(&node.velocity, particle.velocity*particle.mass*w);
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
 * MaterialConstants -- Global parameters dealing with the physics of the world
 * grid -- parameters defining the grid
 *
 * Out:
 * nodes -- updated velocity and velocityChange
 *
 */
__global__ void updateVelocities(ParticleGridNode *nodes, float dt, ImplicitCollider* colliders, int numColliders, MaterialConstants *mat, Grid *grid){
    int nodeIdx = blockIdx.x*blockDim.x + threadIdx.x;
    int gridI, gridJ, gridK;
    gridIndexToIJK(nodeIdx, gridI, gridJ, gridK, grid->dim+1);
    ParticleGridNode &node = nodes[nodeIdx];
    vec3 nodePosition = vec3(gridI, gridJ, gridK)*grid->h + grid->pos;

    node.velocity /= node.mass; //Have to normalize velocity by mass to conserve momentum

    vec3 tmpVelocity = node.velocity + dt*(node.force/node.mass);
    checkForAndHandleCollisions(colliders, numColliders, mat->coeffFriction, nodePosition, tmpVelocity);
    node.velocityChange = tmpVelocity - node.velocity;
    node.velocity = tmpVelocity;
}

/**
 * Updates the grid's nodes for this time step. First computes the mass, velocity and force acting on the grid
 * using a kernel over the particles and then updates the velocity in a second kernel over the grid nodes.
 * @param particleData
 * @param grid
 * @param MaterialConstants
 * @param nodes
 * @param dt
 * @param colliders
 * @param numColliders
 */
void gridMath(Particle *particleData, int numParticles, Grid *grid, MaterialConstants *mat, ParticleGridNode *nodes,
              float dt, ImplicitCollider* colliders, int numColliders, ParticleTempData *devPTGD){
    int threadCount = 256;
    computeParticleGridTempData<<< numParticles / threadCount , threadCount >>>(particleData, grid, mat, devPTGD);
    dim3 blockDim = dim3(numParticles / threadCount / 8, numParticles / threadCount / 8);
    dim3 threadDim = dim3(threadCount/64, 64);
    computeCellMassVelocityAndForceFast<<< blockDim, threadDim >>>(particleData, grid, devPTGD, nodes);
    updateVelocities<<< grid->nodeCount() / 256, 256 >>>(nodes, dt, colliders, numColliders, mat, grid);
}


// Begin testing code:

__global__ void testHalfPlaneColliding(){
    printf("\nTesting half plane colliding:\n");
    ImplicitCollider halfPlane;
    halfPlane.center = vec3(0,0,0);
    halfPlane.param = vec3(0,1,0);
    halfPlane.type = HALF_PLANE;

    if (isColliding(halfPlane, vec3(1,1,1))){ //expect no collision
        printf("\t[FAILED]: Simple non-colliding test on halfplane \n");
    } else{
        printf("\t[PASSED]: Simple non-colliding test on half plane \n");
    }
    if (!isColliding(halfPlane, vec3(-1,-1,-1))){ // expect collision
        printf("\t[FAILED]: Simple colliding test on halfplane failed\n");
    } else{
        printf("\t[PASSED]: Simple colliding test on half plane \n");
    }

    halfPlane.center = vec3(0,10,0);
    halfPlane.param = vec3(1,1,0);
    halfPlane.type = HALF_PLANE;

    if (isColliding(halfPlane, vec3(2,11,1))){ //expect no collision
        printf("\t[FAILED]: Non-colliding test on halfplane \n");
    } else{
        printf("\t[PASSED]: Non-colliding test on half plane \n");
    }
    if (!isColliding(halfPlane, vec3(-1,-1,-1))){ // expect collision
        printf("\t[FAILED]: Colliding test on halfplane \n");
    } else{
        printf("\t[PASSED]: Colliding test on half plane \n");
    }

    printf("Done testing half plane colliding\n\n");
}

__global__ void testSphereColliding(){
    printf("\nTesting sphere colliding:\n");
    ImplicitCollider sphereCollider;
    sphereCollider.center = vec3(0,0,0);
    sphereCollider.param = vec3(1,0,0);
    sphereCollider.type = SPHERE;

    if (isColliding(sphereCollider, vec3(1,1,1))){ //expect no collision
        printf("\t[FAILED]: Simple non-colliding test\n");
    } else{
        printf("\t[PASSED]: Simple non-colliding test\n");
    }
    if (!isColliding(sphereCollider, vec3(.5,0,0))){ // expect collision
        printf("\t[FAILED]: Simple colliding test\n");
    } else{
        printf("\t[PASSED]: Simple colliding test\n");
    }

    sphereCollider.center = vec3(0,10,0);
    sphereCollider.param = vec3(3.2,0,0);
    sphereCollider.type = SPHERE;

    if (isColliding(sphereCollider, vec3(0,0,0))){ //expect no collision
        printf("\t[FAILED]: Non-colliding test \n");
    } else{
        printf("\t[PASSED]: Non-colliding test \n");
    }
    if (!isColliding(sphereCollider, vec3(-1,10,-1))){ // expect collision
        printf("\t[FAILED]: Colliding test\n");
    } else{
        printf("\t[PASSED]: Colliding test\n");
    }


    printf("Done testing sphere colliding\n\n");
}

void testColliding(){
    testHalfPlaneColliding<<<1,1>>>();
    testSphereColliding<<<1,1>>>();
    cudaDeviceSynchronize();
}

__host__ __device__ bool operator==(const vec3 &vecA, const vec3 &vecB)
{
   const double epsilion = 0.0001;  // choose something apprpriate.

   return    std::fabs(vecA[0] -vecB[0]) < epsilion
          && std::fabs(vecA[1] -vecB[1]) < epsilion
          && std::fabs(vecA[2] -vecB[2]) < epsilion;
}

__global__ void testHalfPlaneColliderNormal(){
    printf("\nTesting half plane colliderNormal:\n");
    ImplicitCollider halfPlane;
    halfPlane.center = vec3(0,0,0);
    halfPlane.param = vec3(0,1,0);
    halfPlane.type = HALF_PLANE;


    vec3 normal;
    vec3 expected = vec3(0,1,0);
    colliderNormal(halfPlane, vec3(1,-.001, 1), normal);
    if (normal == expected){
        printf("\t[PASSED]: Simple colliderNormal test on half plane \n");
    } else{
        printf("\t[FAILED]: Simple colliderNormal test on halfplane \n");
    }

    halfPlane.center = vec3(0,10,0);
    halfPlane.param = vec3(1,1,0);
    halfPlane.type = HALF_PLANE;

    expected = vec3(1,1,0);
    colliderNormal(halfPlane, vec3(0,9.999, 0), normal);
    if (expected == normal){
        printf("\t[PASSED]: colliderNormal test on half plane \n");
    } else{
        printf("\t[FAILED]: colliderNormal test on halfplane \n");
    }

    printf("Done testing half plane colliderNormal\n\n");
}

__global__ void testSphereColliderNormal(){
    printf("\nTesting sphere colliderNormal:\n");
    ImplicitCollider sphereCollider;
    sphereCollider.center = vec3(0,0,0);
    sphereCollider.param = vec3(1,0,0);
    sphereCollider.type = SPHERE;

    vec3 normal;
    vec3 expected = vec3::normalize(vec3(1.0f));
    colliderNormal(sphereCollider, vec3(.1f), normal);
    if (normal == expected){
        printf("\t[PASSED]: Simple colliderNormal test\n");
    } else{
        printf("\t[FAILED]: Simple colliderNormal test\n");
    }

    sphereCollider.center = vec3(0,10,0);
    sphereCollider.param = vec3(3.2,0,0);
    sphereCollider.type = SPHERE;

    expected = vec3(0,1,0);
    colliderNormal(sphereCollider, vec3(0,11,0), normal);
    if (normal == expected){
        printf("\t[PASSED]: colliderNormal test \n");
    } else{
       printf("\t[FAILED]: colliderNormal test \n");
    }

    printf("Done testing sphere colliderNormal\n\n");
}

void testColliderNormal(){
    testHalfPlaneColliderNormal<<<1,1>>>();
    testSphereColliderNormal<<<1,1>>>();
    cudaDeviceSynchronize();
}

__device__ float maxNorm(mat3 m){
    float norm = -INFINITY;
    for (int row = 0; row < 3; row++){
        for (int col = 0; col < 3; col++){
            norm=max(norm,fabs(m[col*3+row])); //note glm is column major
        }
    }
    return norm;
}

__device__ void printMat3( const mat3 &mat ) {
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


__global__ void testComputeSigma(){
    printf("\tTesting compute sigma \n");
    Particle p;
    p.mass = 1;
    p.elasticF = mat3(1.0f);
    p.plasticF = mat3(1.0f);
    p.volume = 1;

    MaterialConstants wp;
    wp.mu = 1;
    wp.lambda = 1;
    wp.xi = 1;

    mat3 sigma;
    computeSigma(p, &wp, sigma);

    mat3 expected = mat3(0.0f);

    if (maxNorm(expected-sigma) < 1e-4){
        printf("\t\t[PASSED]: Simple compute sigma\n");
    } else{
        printf("\t\t[FAILED]: Simple compute sigma\n");
    }

    // more complex test
    p.elasticF = mat3(1.f, -1.f, 3.f,
                      3.f, 0.f, 1.f,
                      2.f, 2.f, -1.f);
    p.plasticF = mat3(1.0f);

    computeSigma(p, &wp, sigma);

    expected = mat3( 130.7990,    4.5751,    6.5458,
                     4.5751,  116.4363,  -7.7044,
                     6.5458,   -7.7044,  125.9489);

    if (maxNorm(expected-sigma) < 1e-4){
        printf("\t\t[PASSED]: Complex compute sigma\n");
    } else{
        printf("\t\t[FAILED]: Complex compute sigma\n");
    }

    // even more complex test

    p.elasticF = mat3(0.6062,  0.3500, 0,
                           -0.3500, 0.6062, 0,
                           0,       0,      0.7000);
    p.plasticF = mat3( 0.9000,  0,  0,
                            0, 0.6364, 0.6364,
                            0,-0.6364, 0.6364);

    computeSigma(p, &wp, sigma);

    expected = mat3( -0.8462,    0.0000,         0,
                     0.0000,   -0.8462,        0,
                          0,         0,   -0.8462);

    if (maxNorm(expected-sigma) < 1e-4){
        printf("\t\t[PASSED]: More complex compute sigma\n");
    } else{
        printf("\t\t[FAILED]: More complex compute sigma\n");
    }

    printf("\tDone testing compute sigma \n");
}



__global__ void testCheckForAndHandleCollisions(){
    printf("\tTesting checkAndHandleCollisions\n");
    vec3 position = vec3(0,0,0);
    vec3 velocity = vec3(0,-1,-1);

    ImplicitCollider floor;
    floor.center = vec3(0,0,0);
    floor.param = vec3(0,1,0);
    floor.type = HALF_PLANE;

    ImplicitCollider colliders[1] = {floor};

    float coeffFriction = .5;

    checkForAndHandleCollisions(colliders, 1, coeffFriction, position, velocity);
    vec3 expected = vec3(0,0,-.5);

    if (velocity == expected){
        printf("\t\t[PASSED]: Simple checkAndHandleCollisions test\n");
    } else{
        printf("\t\t[FAILED]: Simple checkAndHandleCollisions test\n");
        printf("\t\t\tActual: (%f, %f, %f)  Expected: (%f, %f, %f)\n", velocity.x, velocity.y, velocity.z, expected.x, expected.y, expected.z);
    }

    velocity = vec3(0,-1,-1);
    coeffFriction = 100000;
    checkForAndHandleCollisions(colliders, 1, coeffFriction, position, velocity);
    expected = vec3(0,0,0);

    if (velocity == expected){
        printf("\t\t[PASSED]: Simple high friction checkAndHandleCollisions test\n");
    } else{
        printf("\t\t[FAILED]: Simple high friction checkAndHandleCollisions test\n");
        printf("\t\t\tActual: (%f, %f, %f)  Expected: (%f, %f, %f)\n", velocity.x, velocity.y, velocity.z, expected.x, expected.y, expected.z);
    }

    ImplicitCollider sphere;
    sphere.center = vec3(0,5,0);
    sphere.param = vec3(1,1,0);
    sphere.type = SPHERE;

    ImplicitCollider colliders2[2] = {floor, sphere};

    position = vec3(0,4,0);
    velocity = vec3(.5,1,-1);
    coeffFriction = .5;
    checkForAndHandleCollisions(colliders2, 2, coeffFriction, position, velocity);
    expected = vec3(.2764,0,-.5528);

    if (velocity == expected){
        printf("\t\t[PASSED]: Simple multiple colliders checkAndHandleCollisions test\n");
    } else{
        printf("\t\t[FAILED]: Simple multiple colliders checkAndHandleCollisions test\n");
        printf("\t\t\tActual: (%f, %f, %f)  Expected: (%f, %f, %f)\n", velocity.x, velocity.y, velocity.z, expected.x, expected.y, expected.z);
    }

    position = vec3(0,4,0);
    velocity = vec3(.5,-1,-1);
    coeffFriction = .5;
    checkForAndHandleCollisions(colliders2, 2, coeffFriction, position, velocity);
    expected = vec3(.5,-1,-1);

    if (velocity == expected){
        printf("\t\t[PASSED]: Simple bodies are separating checkAndHandleCollisions test\n");
    } else{
        printf("\t\t[FAILED]: Simple bodies are separating checkAndHandleCollisions test\n");
        printf("\t\t\tActual: (%f, %f, %f)  Expected: (%f, %f, %f)\n", velocity.x, velocity.y, velocity.z, expected.x, expected.y, expected.z);
    }

    position = vec3(0,100,0);
    velocity = vec3(.5,-1,-1);
    coeffFriction = .5;
    checkForAndHandleCollisions(colliders2, 2, coeffFriction, position, velocity);
    expected = vec3(.5,-1,-1);

    if (velocity == expected){
        printf("\t\t[PASSED]: Simple no collision checkAndHandleCollisions test\n");
    } else{
        printf("\t\t[FAILED]: Simple no collision checkAndHandleCollisions test\n");
        printf("\t\t\tActual: (%f, %f, %f)  Expected: (%f, %f, %f)\n", velocity.x, velocity.y, velocity.z, expected.x, expected.y, expected.z);
    }

    printf("\tDone testing checkAndHandleCollisions\n");
    
}

#define NUM_PARTICLES 2

void testComputeCellMassVelocityAndForceComplex(){
    Particle particles[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; i++){
        particles[i].mass = i+1;
        particles[i].elasticF = mat3(1.0f);//Won't try more complicated values because we test sigma computation elsewhere
        particles[i].plasticF = mat3(1.0f);
        particles[i].velocity = vec3(i+1);
        particles[i].position = vec3(i);
        particles[i].volume = i+1;
    }

    Grid grid;
    grid.dim = glm::ivec3(5,5,5);
    grid.h = 1;
    grid.pos = vec3(-2,-2,-2);

    MaterialConstants wp;
    wp.lambda = wp.mu = wp.xi = 1;

    ParticleGridNode nodes[grid.nodeCount()];

    Particle *dev_particles;
    Grid *dev_grid;
    MaterialConstants *dev_wp;
    ParticleGridNode *dev_nodes;
    ParticleTempData *devPTGD;

    checkCudaErrors(cudaMalloc((void**) &dev_particles, NUM_PARTICLES*sizeof(Particle)));
    checkCudaErrors(cudaMemcpy(dev_particles,particles,NUM_PARTICLES*sizeof(Particle),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_grid, sizeof(Grid)));
    checkCudaErrors(cudaMemcpy(dev_grid,&grid,sizeof(Grid),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_wp, sizeof(MaterialConstants)));
    checkCudaErrors(cudaMemcpy(dev_wp,&wp,sizeof(MaterialConstants),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_nodes, grid.nodeCount()*sizeof(ParticleGridNode)));
    checkCudaErrors(cudaMemcpy(dev_nodes,&nodes,grid.nodeCount()*sizeof(ParticleGridNode),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc( &devPTGD, NUM_PARTICLES*sizeof(ParticleTempData)));



//    computeCellMassVelocityAndForce<<<NUM_PARTICLES, 1>>>(dev_particles, dev_grid, dev_wp, dev_nodes);

    computeParticleGridTempData<<< NUM_PARTICLES, 1 >>>(dev_particles, dev_grid, dev_wp, devPTGD);
    dim3 blockDim = dim3(NUM_PARTICLES);
    dim3 threadDim = dim3(1, 64);
    computeCellMassVelocityAndForceFast<<< blockDim, threadDim >>>(dev_particles, dev_grid, devPTGD, dev_nodes);

    checkCudaErrors(cudaMemcpy(nodes,dev_nodes,grid.nodeCount()*sizeof(ParticleGridNode),cudaMemcpyDeviceToHost));

    //I only check masses because the rest are derived from the same way mass is. The only one that is different is
    // force which I check the sigma function separately
    //These values are from the computeMasses.m file with this initial setup
    float expectedMasses[] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00462962962962964,0.0185185185185185,0.00462962962962964,0,0,0,0.0185185185185185,0.0740740740740741,0.0185185185185185,0,0,0,0.00462962962962964,0.0185185185185185,0.00462962962962964,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0185185185185185,0.0740740740740741,0.0185185185185185,0,0,0,0.0740740740740741,0.305555555555556,0.111111111111111,0.00925925925925927,0,0,0.0185185185185185,0.111111111111111,0.166666666666667,0.0370370370370371,0,0,0,0.00925925925925927,0.0370370370370371,0.00925925925925927,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00462962962962964,0.0185185185185185,0.00462962962962964,0,0,0,0.0185185185185185,0.111111111111111,0.166666666666667,0.0370370370370371,0,0,0.00462962962962964,0.166666666666667,0.597222222222222,0.148148148148148,0,0,0,0.0370370370370371,0.148148148148148,0.0370370370370371,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00925925925925927,0.0370370370370371,0.00925925925925927,0,0,0,0.0370370370370371,0.148148148148148,0.0370370370370371,0,0,0,0.00925925925925927,0.0370370370370371,0.00925925925925927,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
    bool failed = false;
    for (int i =0; i < grid.nodeCount(); i++){
        int I,J,K;
        gridIndexToIJK(i, I, J, K, grid.dim+1);
        ParticleGridNode node = nodes[i];
        if ( std::abs(expectedMasses[i] - node.mass) > 1e-4){
            printf("\t\tActual mass (%f) didn't equal expected mass (%f) for node: (%d, %d, %d)\n", node.mass, expectedMasses[i], I,J,K);
            failed = true;
        }
        //printf("Node: ( %d, %d, %d), mass: %f\n", I,J,K, node.mass);
    }
    if (failed){
        printf("\t\t[FAILED]: Complex computeCellMassVelocityAndForce() test\n");
    }else{
        printf("\t\t[PASSED]: Complex computeCellMassVelocityAndForce() test\n");
    }


    cudaFree(dev_particles);
    cudaFree(dev_grid);
    cudaFree(dev_wp);
    cudaFree(dev_nodes);

    checkCudaErrors(cudaDeviceSynchronize());
}

#define NUM_PARTICLES 2

void testComputeCellMassVelocityAndForce(){
    printf("\tTesting computeCellMassVelocityAndForce()\n");
    Particle particles[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; i++){
        particles[i].mass = i+1;
        particles[i].elasticF = mat3(1.0f);//Won't try more complicated values because we test sigma computation elsewhere
        particles[i].plasticF = mat3(1.0f);
        particles[i].velocity = vec3(i+1);
        particles[i].position = vec3(i);
        particles[i].volume = i+1;
    }

    Grid grid;
    grid.dim = glm::ivec3(1,1,1);
    grid.h = 1;
    grid.pos = vec3(0,0,0);

    MaterialConstants wp;
    wp.lambda = wp.mu = wp.xi = 1;

    ParticleGridNode nodes[grid.nodeCount()];

    Particle *dev_particles;
    Grid *dev_grid;
    MaterialConstants *dev_wp;
    ParticleGridNode *dev_nodes;
    ParticleTempData *devPTGD;

    checkCudaErrors(cudaMalloc((void**) &dev_particles, NUM_PARTICLES*sizeof(Particle)));
    checkCudaErrors(cudaMemcpy(dev_particles,particles,NUM_PARTICLES*sizeof(Particle),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_grid, sizeof(Grid)));
    checkCudaErrors(cudaMemcpy(dev_grid,&grid,sizeof(Grid),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_wp, sizeof(MaterialConstants)));
    checkCudaErrors(cudaMemcpy(dev_wp,&wp,sizeof(MaterialConstants),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_nodes, grid.nodeCount()*sizeof(ParticleGridNode)));
    checkCudaErrors(cudaMemcpy(dev_nodes,&nodes,grid.nodeCount()*sizeof(ParticleGridNode),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc( &devPTGD, NUM_PARTICLES*sizeof(ParticleTempData)));



    //computeCellMassVelocityAndForce<<<NUM_PARTICLES, 1>>>(dev_particles, dev_grid, dev_wp, dev_nodes);

    computeParticleGridTempData<<< NUM_PARTICLES, 1 >>>(dev_particles, dev_grid, dev_wp, devPTGD);
    dim3 blockDim = dim3(NUM_PARTICLES);
    dim3 threadDim = dim3(1, 64);
    computeCellMassVelocityAndForceFast<<< blockDim, threadDim >>>(dev_particles, dev_grid, devPTGD, dev_nodes);

    checkCudaErrors(cudaMemcpy(nodes,dev_nodes,grid.nodeCount()*sizeof(ParticleGridNode),cudaMemcpyDeviceToHost));

    //I only check masses because the rest are derived from the same way mass is. The only one that is different is
    // force which I check the sigma function separately
    //These values are from the computeMasses.m file with this initial setup
    float expectedMasses[8] ={.3056, .1111, .1111, .1667, .1111, .1667, .1667, .5972};
    bool failed = false;
    for (int i =0; i < grid.nodeCount(); i++){
        int I,J,K;
        gridIndexToIJK(i, I, J, K, grid.dim+1);
        ParticleGridNode node = nodes[i];
        if ( std::abs(expectedMasses[i] - node.mass) > 1e-4){
            printf("\t\tActual mass (%f) didn't equal expected mass (%f) for node: (%d, %d, %d)\n", node.mass, expectedMasses[i], I,J,K);
            failed = true;
        }
        //printf("Node: ( %d, %d, %d), mass: %f\n", I,J,K, node.mass);
    }
    if (failed){
        printf("\t\t[FAILED]: Simple computeCellMassVelocityAndForce() test\n");
    }else{
        printf("\t\t[PASSED]: Simple computeCellMassVelocityAndForce() test\n");
    }


    cudaFree(dev_particles);
    cudaFree(dev_grid);
    cudaFree(dev_wp);
    cudaFree(dev_nodes);

    checkCudaErrors(cudaDeviceSynchronize());

    testComputeCellMassVelocityAndForceComplex();
    printf("\tDone testing computeCellMassVelocityAndForce()\n");
}


void testUpdateVelocities(){
    printf("\tTesting updateVelocities\n");

    Grid grid;
    grid.dim = glm::ivec3(1,0,0);
    grid.h = 1;

    float dt = 1;

    ParticleGridNode nodes2[2];
    //nodes[0].position = vec3(0,0,0), implicitly by index
    nodes2[0].mass = 1;
    nodes2[0].force = vec3(1,1,1);
    nodes2[0].velocity = vec3(0,1,0);

    //nodes[1].position = vec3(1,0,0), implicitly by index
    nodes2[1].mass = 2;
    nodes2[1].force = vec3(0,4,0);
    nodes2[1].velocity = vec3(0,0,0);


    ImplicitCollider sphere;
    sphere.center = vec3(1,1,0);
    sphere.param = vec3(1,0,0);
    sphere.type = SPHERE;
    ImplicitCollider colliders[1] = {sphere};

    MaterialConstants wp;
    wp.coeffFriction = .5;

    ImplicitCollider *dev_colliders;
    Grid *dev_grid;
    MaterialConstants *dev_wp;
    ParticleGridNode *dev_nodes;

    checkCudaErrors(cudaMalloc((void**) &dev_colliders, sizeof(ImplicitCollider)));
    checkCudaErrors(cudaMemcpy(dev_colliders,&colliders,sizeof(ImplicitCollider),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_grid, sizeof(Grid)));
    checkCudaErrors(cudaMemcpy(dev_grid,&grid,sizeof(Grid),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_wp, sizeof(MaterialConstants)));
    checkCudaErrors(cudaMemcpy(dev_wp,&wp,sizeof(MaterialConstants),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_nodes, grid.nodeCount()*sizeof(ParticleGridNode)));
    checkCudaErrors(cudaMemcpy(dev_nodes,&nodes2,grid.nodeCount()*sizeof(ParticleGridNode),cudaMemcpyHostToDevice));

    updateVelocities<<<2,1>>>(dev_nodes, dt, dev_colliders, 1, dev_wp, dev_grid);

    checkCudaErrors(cudaMemcpy(nodes2,dev_nodes,grid.nodeCount()*sizeof(ParticleGridNode),cudaMemcpyDeviceToHost));

    vec3 node0VExpected = vec3(1,2,1);
    if (nodes2[0].velocity == node0VExpected){
        printf("\t\t[PASSED]: Simple no collision updateVelocities test\n");
    } else{
        printf("\t\t[FAILED]: Simple no collision updateVelocities test\n");
        vec3 expected = node0VExpected;
        vec3 velocity = nodes2[0].velocity;
        printf("\t\t\tActual: (%f, %f, %f)  Expected: (%f, %f, %f)\n", velocity.x, velocity.y, velocity.z, expected.x, expected.y, expected.z);
    }

    vec3 node1VExpected = vec3(0,0,0);
    if (nodes2[1].velocity == node1VExpected){
        printf("\t\t[PASSED]: Simple no collision updateVelocities test\n");
    } else{
        printf("\t\t[FAILED]: Simple no collision updateVelocities test\n");
        vec3 expected = node1VExpected;
        vec3 velocity = nodes2[1].velocity;
        printf("\t\t\tActual: (%f, %f, %f)  Expected: (%f, %f, %f)\n", velocity.x, velocity.y, velocity.z, expected.x, expected.y, expected.z);
    }

    cudaFree(dev_colliders);
    cudaFree(dev_grid);
    cudaFree(dev_wp);
    cudaFree(dev_nodes);

    printf("\tDone testing updateVelocities");
}

void testGridMath(){
    printf("\nTesting grid math\n");
    testComputeSigma<<<1,1>>>();
    cudaDeviceSynchronize();

    testComputeCellMassVelocityAndForce();

    testCheckForAndHandleCollisions<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    testUpdateVelocities();


    printf("Done testing grid math\n");

}

void timingTests(){
    const int dim = 128;
    Grid grid;
    grid.dim = glm::ivec3( dim, dim, dim );
    grid.h = 1.f/dim;
    grid.pos = vec3(0,0,0);

    int nParticles = 5000*32;
    printf( "    Generating %d particles (%.2f MB)...\n",
            nParticles, nParticles*sizeof(Particle)/1e6 );
    fflush(stdout);
    Particle *particles = new Particle[nParticles];
    for ( int i = 0; i < nParticles; ++i ) {
        Particle particle;
        particle.position = grid.pos + vec3( urand(), urand(), urand() );
        particle.velocity = vec3( 0.f, -0.124f, 0.f );
        particle.elasticF = mat3(1.f);
        particle.plasticF = mat3(1.f);
        particles[i] = particle;
    }

    printf( "    Generating %d grid nodes (%.2f MB)...\n",
            (dim+1)*(dim+1)*(dim+1), (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGridNode)/1e6 );
    fflush(stdout);
    ParticleGridNode *nodes = new ParticleGridNode[grid.nodeCount()];
    for ( int i = 0; i <= dim; ++i ) {
        for ( int j = 0; j <= dim; ++j ) {
            for ( int k = 0; k <= dim; ++k ) {
                ParticleGridNode node;
                node.velocity = vec3( 0.f, 0.f, 0.f );
                node.velocityChange = vec3( 0.f, 0.f, 0.f );
                nodes[i*(dim+1)*(dim+1)+j*(dim+1)+k] = node;
            }
        }
    }

    MaterialConstants mat;
    mat.mu = 58333;
    mat.lambda = 38888;
    mat.xi = 10;
    mat.coeffFriction = .1;

    ImplicitCollider floor;
    floor.center = vec3(0,0,0);
    floor.param = vec3(0,1,0);
    floor.type = HALF_PLANE;

    ImplicitCollider colliders[] = {floor};
    int nColliders = 1;

    printf( "    Allocating kernel resources...\n" ); fflush(stdout);
    Particle *devParticles;
    ParticleGridNode *devNodes;
    Grid *devGrid;
    MaterialConstants *devMaterialConstants;
    ImplicitCollider *devColliders;
    ParticleTempData *devPTGD;
    checkCudaErrors(cudaMalloc( &devParticles, nParticles*sizeof(Particle) ));
    checkCudaErrors(cudaMalloc( &devNodes, (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGridNode) ));
    checkCudaErrors(cudaMalloc( &devGrid, sizeof(Grid) ));
    checkCudaErrors(cudaMalloc( &devMaterialConstants, sizeof(mat) ));
    checkCudaErrors(cudaMalloc( &devColliders, nColliders*sizeof(ImplicitCollider) ));
    checkCudaErrors(cudaMalloc( &devPTGD, nParticles*sizeof(ParticleTempData)));

    static const int blockSizes[] = { 64, 128, 256, 512 };
    static const int nBlocks = 4;

    float dt = .001;
    for ( int i = 0; i < nBlocks; ++i ) {
        checkCudaErrors(cudaMemcpy( devParticles, particles, nParticles*sizeof(Particle), cudaMemcpyHostToDevice ));
        checkCudaErrors(cudaMemcpy( devNodes, nodes, (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGridNode), cudaMemcpyHostToDevice ));
        checkCudaErrors(cudaMemcpy( devGrid, &grid, sizeof(Grid), cudaMemcpyHostToDevice ));
        checkCudaErrors(cudaMemcpy( devMaterialConstants, &mat, sizeof(mat), cudaMemcpyHostToDevice ));
        checkCudaErrors(cudaMemcpy( devColliders, colliders, nColliders*sizeof(ImplicitCollider), cudaMemcpyHostToDevice ));
        int threadCount = blockSizes[i];
//        int blockSize = 256;
        printf( "    Block size = %3d; ", threadCount ); fflush(stdout);

        TIME( " Launching full kernel... ", "finished\n",

              computeParticleGridTempData<<< nParticles / threadCount , threadCount >>>(devParticles, devGrid, devMaterialConstants, devPTGD);
              dim3 blockDim = dim3(nParticles / threadCount, 64);
              dim3 threadDim = dim3(threadCount/64, 64);
              computeCellMassVelocityAndForceFast<<< blockDim, threadDim >>>(devParticles, devGrid, devPTGD, devNodes);
//              computeCellMassVelocityAndForce<<< nParticles / threadCount, threadCount >>>(devParticles, devGrid, devMaterialConstants, devNodes);
              updateVelocities<<< grid.nodeCount() / threadCount, threadCount >>>(devNodes, dt, devColliders, nColliders, devMaterialConstants, devGrid);
              checkCudaErrors(cudaDeviceSynchronize());
        );
    }

//    checkCudaErrors(cudaMemcpy(nodes,devNodes,grid.nodeCount()*sizeof(ParticleGridNode),cudaMemcpyDeviceToHost));
//    printf("mass: %f", nodes[21489].mass);

    printf( "    Freeing kernel resources...\n" ); fflush(stdout);
    checkCudaErrors(cudaFree( devParticles ));
    checkCudaErrors(cudaFree( devNodes ));
    checkCudaErrors(cudaFree( devGrid ));
    checkCudaErrors(cudaFree( devMaterialConstants ));
    checkCudaErrors(cudaFree( devColliders ));
    checkCudaErrors(cudaFree( devPTGD ));
    delete [] particles;
    delete [] nodes;
}



// IT IS SLOW!
void testFillParticleVolume(){
    int numParticles = 128;
    Particle *particles = new Particle[numParticles];

    for (int i = 0; i < numParticles; i++){
        particles[i].mass = 1;
        particles[i].position = vec3(urand(), urand(), urand());
        particles[i].volume = 0;
    }

    Grid grid;
    int dim = 16;
    grid.dim = glm::ivec3(dim,dim,dim);
    grid.h = 1.0f/dim;

    float *cellMasses = new float[grid.nodeCount()];
    memset(cellMasses, 0, sizeof(float)*grid.nodeCount());

    printf("Computing cell masses...\n"); fflush(stdout);

    //Compute cell masses
    for (int nIdx = 0; nIdx < grid.nodeCount(); nIdx++){
        for (int pIdx = 0; pIdx < numParticles; pIdx++){
            Particle &particle = particles[pIdx];

            glm::ivec3 IJK;
            gridIndexToIJK(nIdx, grid.nodeDim(), IJK);


            vec3 nodePosition(IJK.x, IJK.y, IJK.z);
            vec3 particleGridPos = (particle.position - grid.pos)/grid.h;
            vec3 dx = vec3::abs(particleGridPos-nodePosition);
            float w = weight(dx);
            if (particle.mass > 1 || w > 10){
                printf("mass: %f, w: %f", particle.mass, w);
            }
            cellMasses[nIdx] += w*particle.mass;
        }
    }

    for (int i =0 ; i < grid.nodeCount(); i++){
        printf("cellMasses[i]: %f\n", cellMasses[i]);
    }

    printf("Computing particle volumes... \n"); fflush(stdout);

    //Compute volumes
    float *volumes = new float[numParticles];
    memset(volumes, 0, sizeof(float)*numParticles);

    for (int pIdx = 0; pIdx < numParticles; pIdx++){
        Particle &particle = particles[pIdx];
        for (int nIdx = 0; nIdx < grid.nodeCount(); nIdx++){
            glm::ivec3 IJK;
            gridIndexToIJK(nIdx, grid.nodeDim(), IJK);


            vec3 nodePosition(IJK.x, IJK.y, IJK.z);
            vec3 particleGridPos = (particle.position - grid.pos)/grid.h;
            vec3 dx = vec3::abs(particleGridPos-nodePosition);
            float w = weight(dx);
            volumes[pIdx] += cellMasses[nIdx]*w;
        }
        float gridVolume = grid.h*grid.h*grid.h;
        volumes[pIdx] = particle.mass / (volumes[pIdx] / gridVolume);
    }

     Particle *devParticles;
     Grid *devGrid;
     checkCudaErrors(cudaMalloc( &devParticles, numParticles*sizeof(Particle) ));
     checkCudaErrors(cudaMemcpy( devParticles, particles, numParticles*sizeof(Particle), cudaMemcpyHostToDevice ));
     checkCudaErrors(cudaMalloc( &devGrid, sizeof(Grid) ));
     checkCudaErrors(cudaMemcpy( devGrid, &grid, sizeof(Grid), cudaMemcpyHostToDevice ));

     printf("Calling fillParticleVolume kernel\n"); fflush(stdout);

     fillParticleVolume(devParticles, numParticles, devGrid, grid.nodeCount());

     printf("Comparing values\n"); fflush(stdout);

     checkCudaErrors(cudaMemcpy( particles, devParticles, numParticles*sizeof(Particle), cudaMemcpyDeviceToHost ));

     bool failed = false;
     for (int i = 0; i < numParticles; i++){
         if (std::fabs(volumes[i] - particles[i].volume) > 1e-8 || std::isnan(particles[i].volume)){
             failed = true;
//             printf("Expected: %f, Actual: %f\n", volumes[i], particles[i].volume); fflush(stdout);
         }
         printf("volume: %g, part.volume: %g\n", volumes[i], particles[i].volume);
     }

     if (failed){
         printf("[FAILED]: test fillParticleVolume Test\n");
     } else{
         printf("[PASSED]: test fillParticleVolume Test\n");
     }

     checkCudaErrors(cudaFree( devParticles ));
     checkCudaErrors(cudaFree( devGrid ));
     delete[] cellMasses;
     delete[] volumes;
}


#define NUM_PARTICLES 2
void testcomputeCellMasses(){
    Particle particles[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; i++){
        particles[i].mass = i+1;
        particles[i].elasticF = mat3(1.0f);//Won't try more complicated values because we test sigma computation elsewhere
        particles[i].plasticF = mat3(1.0f);
        particles[i].velocity = vec3(i+1);
        particles[i].position = vec3(i);
        particles[i].volume = 0;
    }

    Grid grid;
    grid.dim = glm::ivec3(5,5,5);
    grid.h = 1;
    grid.pos = vec3(-2,-2,-2);

    Particle *dev_particles;
    Grid *dev_grid;
    float *devCellMasses;

    checkCudaErrors(cudaMalloc((void**) &dev_particles, NUM_PARTICLES*sizeof(Particle)));
    checkCudaErrors(cudaMemcpy(dev_particles,particles,NUM_PARTICLES*sizeof(Particle),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_grid, sizeof(Grid)));
    checkCudaErrors(cudaMemcpy(dev_grid,&grid,sizeof(Grid),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc( &devCellMasses, grid.nodeCount()*sizeof(float)));
    cudaMemset(devCellMasses, 0, grid.nodeCount()*sizeof(float));


    dim3 blockDim = dim3(NUM_PARTICLES);
    dim3 threadDim = dim3(1, 64);
    computeCellMasses<<< blockDim, threadDim >>>(dev_particles, dev_grid, devCellMasses);

    float *cellMasses = new float[grid.nodeCount()];

    checkCudaErrors(cudaMemcpy(cellMasses,devCellMasses,grid.nodeCount()*sizeof(float),cudaMemcpyDeviceToHost));

    //I only check masses because the rest are derived from the same way mass is. The only one that is different is
    // force which I check the sigma function separately
    //These values are from the computeMasses.m file with this initial setup
    float expectedMasses[] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00462962962962964,0.0185185185185185,0.00462962962962964,0,0,0,0.0185185185185185,0.0740740740740741,0.0185185185185185,0,0,0,0.00462962962962964,0.0185185185185185,0.00462962962962964,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0185185185185185,0.0740740740740741,0.0185185185185185,0,0,0,0.0740740740740741,0.305555555555556,0.111111111111111,0.00925925925925927,0,0,0.0185185185185185,0.111111111111111,0.166666666666667,0.0370370370370371,0,0,0,0.00925925925925927,0.0370370370370371,0.00925925925925927,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00462962962962964,0.0185185185185185,0.00462962962962964,0,0,0,0.0185185185185185,0.111111111111111,0.166666666666667,0.0370370370370371,0,0,0.00462962962962964,0.166666666666667,0.597222222222222,0.148148148148148,0,0,0,0.0370370370370371,0.148148148148148,0.0370370370370371,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00925925925925927,0.0370370370370371,0.00925925925925927,0,0,0,0.0370370370370371,0.148148148148148,0.0370370370370371,0,0,0,0.00925925925925927,0.0370370370370371,0.00925925925925927,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
    bool failed = false;
    for (int i =0; i < grid.nodeCount(); i++){
        int I,J,K;
        gridIndexToIJK(i, I, J, K, grid.dim+1);
        if ( std::abs(expectedMasses[i] - cellMasses[i]) > 1e-4 || std::isnan(cellMasses[i])){
            printf("\t\tActual mass (%f) didn't equal expected mass (%f) for node: (%d, %d, %d), %d\n", cellMasses[i], expectedMasses[i], I,J,K, i); fflush(stdout);
            failed = true;
        }
//        printf("Node: ( %d, %d, %d), mass: %f, expected: %f\n", I,J,K, cellMasses[i], expectedMasses[i]);
    }
    if (failed){
        printf("\t\t[FAILED]: Complex computeCellMasses() test\n");
    }else{
        printf("\t\t[PASSED]: Complex computeCellMassse() test\n");
    }
checkCudaErrors(cudaDeviceSynchronize());

    computeParticleDensity<<< blockDim, threadDim >>>(dev_particles, dev_grid, devCellMasses);

    checkCudaErrors(cudaMemcpy(particles,dev_particles,NUM_PARTICLES*sizeof(Particle),cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < NUM_PARTICLES; i++){
        printf("density: %f\n", particles[i].volume);
    }

    computeParticleVolume<<< NUM_PARTICLES, 1 >>>(dev_particles, dev_grid);

    checkCudaErrors(cudaMemcpy(particles,dev_particles,NUM_PARTICLES*sizeof(Particle),cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < NUM_PARTICLES; i++){
        printf("volumes: %f\n", particles[i].volume);
    }

    cudaFree(dev_particles);
    cudaFree(dev_grid);
    cudaFree(devCellMasses);

    delete[] cellMasses;

    checkCudaErrors(cudaDeviceSynchronize());

//    printf("\tDone testing computeCellMassVelocityAndForce()\n");

}



#endif // WIL_CU
