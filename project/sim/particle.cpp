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
#include "common/common.h"
#include "cuda/functions.h"

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
{    /**
     * adds snow container OBJ to XML tree
     */
    void addSnowVolume(QDomElement &node);

    /**
     * adds <medium> tag to XML tree. Also calls exportVolumeData to write out volume.
     * calls exportVolumeData, then if successful, links to the .vol file
     */
    void writeMedium(QDomElement &node);
    if ( !hasVBO() ) buildVBO();

    glPushAttrib( GL_LIGHTING_BIT );
    glDisable( GL_LIGHTING );
    glColor3f( 0.8f, 0.8f, 1.f );
    glPointSize( 1.f );

    glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, sizeof(Particle), (void*)(0) );
    glDrawArrays( GL_POINTS, 0, m_particles.size() );
    glDisableClientState( GL_VERTEX_ARRAY );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    glPopAttrib();
}

void
ParticleSystem::update( float time )
{
    if ( !hasVBO() ) buildVBO();
    updateParticles( &m_cudaVBO, time, m_particles.size() );
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
    registerVBO( &m_cudaVBO, m_glVBO );
}

void
ParticleSystem::deleteVBO()
{
    // Delete OpenGL VBO and unregister with CUDA
    if ( hasVBO() ) {
        unregisterVBO( m_cudaVBO );
        glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
        glDeleteBuffers( 1, &m_glVBO );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
    }
    m_glVBO = 0;
}
