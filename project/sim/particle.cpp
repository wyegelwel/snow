/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   particle.cpp
**   Author: mliberma
**   Created: 6 Apr 2014
**
**************************************************************************/

#include "particle.h"

#include <GL/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "common/common.h"

extern "C"
void updateCUDA( float time, Particle *particles, int nParticles );

ParticleSystem::ParticleSystem()
{
    m_glVBO = 0;
}

ParticleSystem::~ParticleSystem()
{
    deleteVBO();
}

void
ParticleSystem::clear()
{
    m_particles.clear();
    deleteVBO();
}

void
ParticleSystem::render()
{
    if ( !hasVBO() ) buildVBO();

    glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, sizeof(Particle), (void*)(0) );
    glDrawArrays( GL_POINTS, 0, m_particles.size() );
    glDisableClientState( GL_VERTEX_ARRAY );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
}

void
ParticleSystem::update( float time )
{
    if ( !hasVBO() ) buildVBO();

    checkCudaErrors( cudaGraphicsMapResources(1, &m_cudaVBO, 0) );

    Particle *devPtr;
    size_t size;
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, m_cudaVBO) );

    updateCUDA( time, devPtr, m_particles.size() );

    checkCudaErrors( cudaGraphicsUnmapResources(1, &m_cudaVBO, 0) );
}

bool
ParticleSystem::hasVBO() const
{
    return m_glVBO > 0 && glIsBuffer( m_glVBO );
}

void
ParticleSystem::buildVBO()
{
    deleteVBO();

    // Build OpenGL VBO
    glGenBuffers( 1, &m_glVBO );
    glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
    glBufferData( GL_ARRAY_BUFFER, m_particles.size()*sizeof(Particle), m_particles.data(), GL_DYNAMIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    // Register OpenGL VBO with CUDA
    checkCudaErrors( cudaGraphicsGLRegisterBuffer(&m_cudaVBO, m_glVBO, cudaGraphicsMapFlagsWriteDiscard) );
}

void
ParticleSystem::deleteVBO()
{
    // Delete OpenGL VBO
    if ( hasVBO() ) {
        cudaGraphicsUnregisterResource( m_cudaVBO );
        glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
        glDeleteBuffers( 1, &m_glVBO );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
    }
    m_glVBO = 0;
}
