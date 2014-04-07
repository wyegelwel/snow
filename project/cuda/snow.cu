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
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <glm/glm.hpp>

struct Particle
{
    glm::vec3 position;
    glm::vec3 velocity;
    float mass;
    float volume;
    glm::mat3 elasticF;
    glm::mat3 plasticF;
};

extern "C"
void updateCUDA( float time, Particle *particles, int particleCount );

__global__ void snow_kernel( float time, Particle *particles )
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    glm::vec3 pn = glm::normalize( particles[index].position );
    particles[index].position += 0.05f*sinf(10*time)*pn;
}

void updateCUDA( float time, Particle *particles, int particleCount )
{
    snow_kernel<<< particleCount/512, 512 >>>( time, particles );
}

#endif // SNOW_CU
