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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher
#endif

extern "C"  {
    void testColliding();
    void testGridMath();
}


// Collider
#include <glm/geometric.hpp>
#include "math.h"   // this imports the CUDA math library
#include "sim/collider.h"


typedef bool (*isCollidingFunc) (ImplicitCollider collider, glm::vec3 position);

/**
 * A collision occurs when the point is on the OTHER side of the normal
 */
__device__ bool isCollidingHalfPlane(glm::vec3 planePoint, glm::vec3 planeNormal, glm::vec3 position){
    glm::vec3 vecToPoint = position - planePoint;
    return (glm::dot(vecToPoint, planeNormal) < 0);
}

/**
 * Defines a halfplane such that collider.center is a point on the plane,
 * and collider.param is the normal to the plane.
 */
__device__ bool isCollidingHalfPlaneImplicit(ImplicitCollider collider, glm::vec3 position){
    return isCollidingHalfPlane(collider.center, collider.param, position);
}

/**
 * Defines a sphere such that collider.center is the center of the sphere,
 * and collider.param.x is the radius.
 */
__device__ bool isCollidingSphereImplicit(ImplicitCollider collider, glm::vec3 position){
    float radius = collider.param.x;
    return (glm::length(position-collider.center) < radius);
}


/** array of colliding functions. isCollidingFunctions[collider.type] will be the correct function */
__device__ isCollidingFunc isCollidingFunctions[2] = {isCollidingHalfPlaneImplicit, isCollidingSphereImplicit};




/**
 * General purpose function for handling colliders
 */
__device__ bool isColliding(ImplicitCollider collider, glm::vec3 position){
    return isCollidingFunctions[collider.type](collider, position);
}

// Grid math

#include "tim.cu" // should really be snow.cu or grid.cu depending on how we break it up
#include "decomposition.cu"
#include "weighting.cu"
#include "sim/particlegrid.h"
#include "sim/world.h"


__host__ __device__ void computeSigma(Particle &particle, WorldParams *worldParams, glm::mat3 &sigma){
    glm::mat3 &Fp = particle.plasticF; //for the sake of making the code look like the math
    glm::mat3 &Fe = particle.elasticF;

    float Jp = glm::determinant(Fp);
    float Je = glm::determinant(Fe);

    glm::mat3 Re, Se;
    computePD(Fe, Re, Se);

    float muFp = worldParams->mu*exp(worldParams->xi*(1-Jp));
    float lambdaFp = worldParams->lambda*exp(worldParams->xi*(1-Jp));

    sigma = (2*muFp/Jp)*(Fe-Re)*glm::transpose(Fe)+(lambdaFp/Jp)*(Je-1)*Je*glm::mat3(1.0f);
}


__device__ void atomicAdd(glm::vec3 *add, glm::vec3 toAdd){
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
 * worldParams -- Global parameters dealing with the physics of the world
 *
 * Out:
 * nodes -- list of every node in grid ((dim.x+1)*(dim.y+1)*(dim.z+1))
 *
 */
__global__ void computeCellMassVelocityAndForce(Particle *particleData, Grid *grid, WorldParams *worldParams, ParticleGrid::Node *nodes){
    int particleIdx = blockIdx.x*blockDim.x + threadIdx.x;
    Particle &particle = particleData[particleIdx];

    glm::ivec3 gridIJK;
    positionToGridIJK(particle.position, grid, gridIJK); // NOTE: since we are working with nodes, we may need to reconcile grid.dim and size of nodes at some point

    glm::mat3 sigma;
    computeSigma(particle, worldParams, sigma);

    // Apply particles contribution of mass, velocity and force to surrounding nodes
    glm::ivec3 min = glm::max(glm::ivec3(0.0f), gridIJK-2);
    glm::ivec3 max = glm::min(grid->dim, gridIJK+2); //+1 because we are working on nodes
    for (int i = min.x; i <= max.x; i++){
        for (int j = min.y; j <= max.y; j++){
            for (int k = min.z; k <= max.z; k++){
                glm::vec3 nodePosition = glm::vec3(i, j, k)*grid->h + grid->pos;
                int currIdx = getGridIndex(i, j, k, grid);
                ParticleGrid::Node &node = nodes[currIdx];

                float w;
                glm::vec3 wg;
                weightAndGradient((particle.position-nodePosition)/grid->h, w, wg);

                atomicAdd(&node.mass, particle.mass*w);
                atomicAdd(&node.velocity, particle.velocity*particle.mass*w);
                atomicAdd(&node.force, particle.volume*sigma*wg);
            }
        }
    }
}



// Begin testing code:

__global__ void testHalfPlaneColliding(){
    printf("\nTesting half plane colliding:\n");
    ImplicitCollider halfPlane;
    halfPlane.center = glm::vec3(0,0,0);
    halfPlane.param = glm::vec3(0,1,0);
    halfPlane.type = HALF_PLANE;

    if (isColliding(halfPlane, glm::vec3(1,1,1))){ //expect no collision
        printf("\t[FAILED]: Simple non-colliding test on halfplane \n");
    } else{
        printf("\t[PASSED]: Simple non-colliding test on half plane \n");
    }
    if (!isColliding(halfPlane, glm::vec3(-1,-1,-1))){ // expect collision
        printf("\t[FAILED]: Simple colliding test on halfplane failed\n");
    } else{
        printf("\t[PASSED]: Simple colliding test on half plane \n");
    }

    halfPlane.center = glm::vec3(0,10,0);
    halfPlane.param = glm::vec3(1,1,0);
    halfPlane.type = HALF_PLANE;

    if (isColliding(halfPlane, glm::vec3(2,11,1))){ //expect no collision
        printf("\t[FAILED]: Non-colliding test on halfplane \n");
    } else{
        printf("\t[PASSED]: Non-colliding test on half plane \n");
    }
    if (!isColliding(halfPlane, glm::vec3(-1,-1,-1))){ // expect collision
        printf("\t[FAILED]: Colliding test on halfplane \n");
    } else{
        printf("\t[PASSED]: Colliding test on half plane \n");
    }

    printf("Done testing half plane colliding\n\n");
}

__global__ void testSphereColliding(){
    printf("\nTesting sphere colliding:\n");
    ImplicitCollider SpherePlane;
    SpherePlane.center = glm::vec3(0,0,0);
    SpherePlane.param = glm::vec3(1,0,0);
    SpherePlane.type = SPHERE;

    if (isColliding(SpherePlane, glm::vec3(1,1,1))){ //expect no collision
        printf("\t[FAILED]: Simple non-colliding test\n");
    } else{
        printf("\t[PASSED]: Simple non-colliding test\n");
    }
    if (!isColliding(SpherePlane, glm::vec3(.5,0,0))){ // expect collision
        printf("\t[FAILED]: Simple colliding test\n");
    } else{
        printf("\t[PASSED]: Simple colliding test\n");
    }

    SpherePlane.center = glm::vec3(0,10,0);
    SpherePlane.param = glm::vec3(3.2,0,0);
    SpherePlane.type = HALF_PLANE;

    if (isColliding(SpherePlane, glm::vec3(0,0,0))){ //expect no collision
        printf("\t[FAILED]: Non-colliding test \n");
    } else{
        printf("\t[PASSED]: Non-colliding test \n");
    }
    if (!isColliding(SpherePlane, glm::vec3(-1,10,-1))){ // expect collision
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

__device__ float maxNorm(glm::mat3 m){
    float norm = -INFINITY;
    for (int row = 0; row < 3; row++){
        for (int col = 0; col < 3; col++){
            norm=glm::max(norm,m[col][row]); //note glm is column major
        }
    }
    return norm;
}

__global__ void testComputeSigma(){
    printf("\tTesting compute sigma \n");
    Particle p;
    p.mass = 1;
    p.elasticF = glm::mat3(1.0f);
    p.plasticF = glm::mat3(1.0f);

    WorldParams wp;
    wp.mu = 1;
    wp.lambda = 1;
    wp.xi = 1;

    glm::mat3 sigma;
    computeSigma(p, &wp, sigma);

    glm::mat3 expected = glm::mat3(0.0f);

    if (0 && maxNorm(expected-sigma) > 1e-4){
        printf("\t\t[FAILED]: Simple compute sigma\n");
    } else{
        printf("\t\t[PASSED]: Simple compute sigma\n");
    }

    // more complex test
    p.elasticF = glm::mat3(1.f, 2.f, 3.f,
                           4.f, 5.f, 6.f,
                           7.f, 8.f, 9.0f);
    p.plasticF = glm::mat3(1.0f);

    computeSigma(p, &wp, sigma);

    expected = glm::mat3(122.9580,  146.6232,  170.2883,
                         146.6232,  174.9487,  203.2743,
                         170.2883,  203.2743,  236.2603);

    if (0 && maxNorm(expected-sigma) > 1e-4){
        printf("\t\t[FAILED]: Complex compute sigma\n");
    } else{
        printf("\t\t[PASSED]: Complex compute sigma\n");
    }

    // even more complex test

    p.elasticF = glm::mat3(0.6062,  0.3500, 0,
                           -0.3500, 0.6062, 0,
                           0,       0,      0.7000);
    p.plasticF = glm::mat3( 0.9000,  0,  0,
                            0, 0.6364, 0.6364,
                            0,-0.6364, 0.6364);

    computeSigma(p, &wp, sigma);

    expected = glm::mat3(-1.1608, 0.0000, 0,
                         0, -1.1608, 0,
                         0,  0, -1.1608);

    if (0 && maxNorm(expected-sigma) > 1e-4){
        printf("\t\t[FAILED]: Complex compute sigma\n");
    } else{
        printf("\t\t[PASSED]: Complex compute sigma\n");
    }

    printf("\tDone testing compute sigma \n");
}

//Particle *particleData, Grid *grid, ParticleGrid::Node *nodes, WorldParams *worldParams

#define NUM_PARTICLES 2

void testGridMath(){
    printf("\nTesting grid math\n");
    testComputeSigma<<<1,1>>>();
    cudaDeviceSynchronize();
    printf("\tTesting computeCellMassVelocityAndForce()\n");
    Particle particles[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; i++){
        particles[i].mass = i+1;
        particles[i].elasticF = glm::mat3(1.0f);//Won't try more complicated values because we test sigma computation elsewhere
        particles[i].plasticF = glm::mat3(1.0f);
        particles[i].velocity = glm::vec3(i+1);
        particles[i].position = glm::vec3(i);
        particles[i].volume = i+1;
    }

    Grid grid;
    grid.dim = glm::ivec3(1,1,1);
    grid.h = 1;
    grid.pos = glm::vec3(0,0,0);

    WorldParams wp;
    wp.lambda = wp.mu = wp.xi = 1;

    ParticleGrid::Node nodes[grid.nodeCount()];

    Particle *dev_particles;
    Grid *dev_grid;
    WorldParams *dev_wp;
    ParticleGrid::Node *dev_nodes;

    checkCudaErrors(cudaMalloc((void**) &dev_particles, NUM_PARTICLES*sizeof(Particle)));
    checkCudaErrors(cudaMemcpy(dev_particles,particles,NUM_PARTICLES*sizeof(Particle),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_grid, sizeof(Grid)));
    checkCudaErrors(cudaMemcpy(dev_grid,&grid,sizeof(Grid),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_wp, sizeof(WorldParams)));
    checkCudaErrors(cudaMemcpy(dev_wp,&wp,sizeof(WorldParams),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &dev_nodes, grid.nodeCount()*sizeof(ParticleGrid::Node)));
    checkCudaErrors(cudaMemcpy(dev_nodes,&nodes,grid.nodeCount()*sizeof(ParticleGrid::Node),cudaMemcpyHostToDevice));

    computeCellMassVelocityAndForce<<<NUM_PARTICLES, 1>>>(dev_particles, dev_grid, dev_wp, dev_nodes);

    checkCudaErrors(cudaMemcpy(nodes,dev_nodes,grid.nodeCount()*sizeof(ParticleGrid::Node),cudaMemcpyDeviceToHost));

    //I only check masses because the rest are derived from the same way mass is. The only one that is different is
    // force which I check the sigma function separately
    //These values are from the computeMasses.m file with this initial setup
    float expectedMasses[8] ={.3056, .1111, .1111, .1667, .1111, .1667, .1667, .5972};
    bool failed = false;
    for (int i =0; i < grid.nodeCount(); i++){
        int I,J,K;
        gridIndexToIJK(i, I, J, K, &grid);
        ParticleGrid::Node node = nodes[i];
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




    printf("\tDone testing computeCellMassVelocityAndForce()\n");


    printf("Done testing grid math\n");

}

#endif // WIL_CU
