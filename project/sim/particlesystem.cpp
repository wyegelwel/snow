/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   particlesystem.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 21 Apr 2014
**
**************************************************************************/

#include "sim/particlesystem.h"

#include <GL/glew.h>
#include <GL/gl.h>
#include <QGLShaderProgram>

#include "common/common.h"
#include "geometry/bbox.h"
#include "ui/uisettings.h"

ParticleSystem::ParticleSystem()
{
    m_glVBO = 0;
    m_glVAO = 0;
}

ParticleSystem::~ParticleSystem()
{
    deleteBuffers();
}

void
ParticleSystem::clear()
{
    m_particles.clear();
    deleteBuffers();
}

void
ParticleSystem::render()
{
    if ( !hasBuffers() ) buildBuffers();

    QGLShaderProgram *shader = ParticleSystem::shader();
    if ( shader ) {
        glPushAttrib( GL_VERTEX_PROGRAM_POINT_SIZE );
        glEnable( GL_VERTEX_PROGRAM_POINT_SIZE );
        shader->bind();
        shader->setUniformValue( "mode", UiSettings::showParticlesMode() );
    } else {
        glPushAttrib( GL_LIGHTING_BIT );
        glDisable( GL_LIGHTING );
        glColor3f( 1.0f, 1.0f, 1.0f );
        glPointSize( 1.f );
    }

    glEnable( GL_POINT_SMOOTH );
    glHint( GL_POINT_SMOOTH_HINT, GL_NICEST );

    glBindVertexArray( m_glVAO );
    glDrawArrays( GL_POINTS, 0, m_particles.size() );
    glBindVertexArray( 0 );

    if ( shader ) {
        shader->release();
        glPopAttrib();
    } else {
        glPopAttrib();
    }
}

bool
ParticleSystem::hasBuffers() const
{
    return m_glVBO > 0 && glIsBuffer( m_glVBO );
}

void
ParticleSystem::buildBuffers()
{
    deleteBuffers();

    // Build OpenGL VBO
    glGenBuffers( 1, &m_glVBO );
    glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
    glBufferData( GL_ARRAY_BUFFER, m_particles.size()*sizeof(Particle), m_particles.data(), GL_DYNAMIC_DRAW );

    // Build OpenGL VAO
    glGenVertexArrays( 1, &m_glVAO );
    glBindVertexArray( m_glVAO );

    // Position attribute
    glEnableVertexAttribArray( 0 );
    glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(0) );

    // Velocity attribute
    glEnableVertexAttribArray( 1 );
    glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(sizeof(vec3)) );

    // Mass attribute
    glEnableVertexAttribArray( 2 );
    glVertexAttribPointer( 2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(2*sizeof(vec3)) );

    // Volume attribute
    glEnableVertexAttribArray( 3 );
    glVertexAttribPointer( 3, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(2*sizeof(vec3)+sizeof(GLfloat)) );

    glBindVertexArray( 0 );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
}

void
ParticleSystem::deleteBuffers()
{
    // Delete OpenGL VBO and unregister with CUDA
    if ( hasBuffers() ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
        glDeleteBuffers( 1, &m_glVBO );
        glDeleteVertexArrays( 1, &m_glVAO );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
    }
    m_glVBO = 0;
    m_glVAO = 0;
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

vec3
ParticleSystem::getCentroid( const glm::mat4 &ctm )
{
    vec3 c(0,0,0);
    for ( int i = 0; i < m_particles.size(); ++i ) {
        const vec3 p = m_particles[i].position;
        glm::vec4 point = ctm * glm::vec4( p.x, p.y, p.z, 1.f );
        c += vec3( point.x, point.y, point.z );
    }
    return c / (float)m_particles.size();
}

QGLShaderProgram* ParticleSystem::SHADER = NULL;

QGLShaderProgram*
ParticleSystem::shader()
{
    if ( SHADER == NULL ) {
        const QGLContext *context = QGLContext::currentContext();
        SHADER = new QGLShaderProgram( context );
        if ( !SHADER->addShaderFromSourceFile(QGLShader::Vertex, ":/shaders/particlesystem.vert") ||
             !SHADER->addShaderFromSourceFile(QGLShader::Fragment, ":/shaders/particlesystem.frag") ) {
            LOG( "ParticleSystem::shader() : Compile error: \n%s\n", STR(SHADER->log().trimmed()) );
            SAFE_DELETE( SHADER );
        } else {
            SHADER->bindAttributeLocation( "particlePosition", 0 );
            SHADER->bindAttributeLocation( "particleVelocity", 1 );
            SHADER->bindAttributeLocation( "particleMass", 2 );
            SHADER->bindAttributeLocation( "particleVolume", 3 );
            glBindFragDataLocation( SHADER->programId(), 0, "fragmentColor" );
            if ( !SHADER->link() ) {
                LOG( "ParticleSystem::shader() : Link error: \n%s\n", STR(SHADER->log().trimmed()) );
                SAFE_DELETE( SHADER );
            }
        }
    }
    return SHADER;
}
