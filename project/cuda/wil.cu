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
#include <glm/geometric.hpp>
#include "math.h"   // this imports the CUDA math library
#include "sim/collider.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher
#endif

extern "C"  {
    void testColliding();
}

typedef bool (*isCollidingFunc) (ImplicitCollider collider, glm::vec3 position);


// Apparent glm::dot doesn't work...
__device__ float dot(glm::vec3 v, glm::vec3 w){
    return v.x * w.x + v.y*w.y + v.z*w.z;
}

/**
 * A collision occurs when the point is on the OTHER side of the normal
 */
__device__ bool isCollidingHalfPlane(glm::vec3 planePoint, glm::vec3 planeNormal, glm::vec3 position){
    glm::vec3 vecToPoint = position - planePoint;
    return (dot(vecToPoint, planeNormal) < 0);
}

/**
 * Defines a halfplane such that collider.center is a point on the plane,
 * and collider.param is the normal to the plane.
 */
__device__ bool isCollidingHalfPlaneImplicit(ImplicitCollider collider, glm::vec3 position){
    return isCollidingHalfPlane(collider.center, collider.param, position);
}

/** array of colliding functions. isCollidingFunctions[collider.type] will be the correct function */
__device__ isCollidingFunc isCollidingFunctions[1] = {isCollidingHalfPlaneImplicit};

/**
 * General purpose function for handling colliders
 */
__device__ bool isColliding(ImplicitCollider collider, glm::vec3 position){
    return isCollidingFunctions[collider.type](collider, position);
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
        printf("\t[FAILED]: Colliding test on halfplane failed\n");
    } else{
        printf("\t[PASSED]: Colliding test on half plane \n");
    }


    printf("Done testing half plane colliding\n\n");
}

void testColliding(){
    testHalfPlaneColliding<<<1,1>>>();
    cudaDeviceSynchronize();
}

#endif // WIL_CU
