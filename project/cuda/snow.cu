/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   snow.cu
**   Author: mliberma
**   Created: 7 Apr 2014
**
**************************************************************************/

#ifndef SNOW_CU
#define SNOW_CU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include <glm/geometric.hpp>

#define CUDA_INCLUDE
#include "sim/particle.h"
#include "cuda/functions.h"

void registerVBO( cudaGraphicsResource **resource, GLuint vbo )
{
    checkCudaErrors( cudaGraphicsGLRegisterBuffer(resource, vbo, cudaGraphicsMapFlagsNone) );
}

void unregisterVBO( cudaGraphicsResource *resource )
{
    checkCudaErrors( cudaGraphicsUnregisterResource(resource) );
}

//__global__ void snow_kernel( float time, Particle *particles )
//{
//    int index = blockIdx.x*blockDim.x + threadIdx.x;
//    vec3 pn = vec3::normalize( particles[index].position );
//    particles[index].position += 0.05f*sinf(6*time)*pn;
////    particles[index].position += 0.01f*pn;
//}

//void updateParticles( Particle *particles, float time, int particleCount )
//{
//    snow_kernel<<< particleCount/512, 512 >>>( time, particles );
//    checkCudaErrors( cudaDeviceSynchronize() );
//}

#endif // SNOW_CU
