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
#include "geometry/bbox.h"

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

    glPushAttrib( GL_LIGHTING_BIT );
    glDisable( GL_LIGHTING );
    glColor3f( 1.0f, 0.f, 0.f );
    glPointSize( 1.f );

    glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, sizeof(Particle), (void*)(0) );
    glDrawArrays( GL_POINTS, 0, m_particles.size() );
    glDisableClientState( GL_VERTEX_ARRAY );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    glPopAttrib();
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

}

void
ParticleSystem::deleteVBO()
{
    // Delete OpenGL VBO and unregister with CUDA
    if ( hasVBO() ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
        glDeleteBuffers( 1, &m_glVBO );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
    }
    m_glVBO = 0;
}

BBox
ParticleSystem::getBBox( const glm::mat4 &ctm )
{
    BBox box;
    for ( int i = 0; i < m_particles.size(); ++i ) {
        const vec3 &p = m_particles[i].position;
        glm::vec4 point = ctm * glm::vec4( p.x, p.y, p.z, 1.f );
        box += vec3( point.x, point.y, point.z );
    }
    return box;
}
