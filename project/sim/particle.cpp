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

ParticleSystem::ParticleSystem()
{
    m_vbo = 0;
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
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, sizeof(Particle), (void*)(0) );
    glDrawArrays( GL_POINTS, 0, m_particles.size() );
    glDisableClientState( GL_VERTEX_ARRAY );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
}

bool
ParticleSystem::hasVBO() const
{
    return m_vbo > 0 && glIsBuffer( m_vbo );
}

void
ParticleSystem::buildVBO()
{
    deleteVBO();
    glGenBuffers( 1, &m_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glBufferData( GL_ARRAY_BUFFER, m_particles.size()*sizeof(Particle), m_particles.data(), GL_DYNAMIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
}

void
ParticleSystem::deleteVBO()
{
    if ( hasVBO() ) {
        glDeleteBuffers( 1, &m_vbo );
    }
    m_vbo = 0;
}
