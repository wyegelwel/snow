/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   particlegrid.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 21 Apr 2014
**
**************************************************************************/

#include <GL/glew.h>
#include <GL/gl.h>
#include <QGLShaderProgram>

#include <QVector>

#include "particlegrid.h"

#include "common/common.h"
#include "geometry/bbox.h"
#include "ui/uisettings.h"

ParticleGrid::ParticleGrid()
    : m_size(0),
      m_glIndices(0),
      m_glVBO(0),
      m_glVAO(0)
{
}

ParticleGrid::~ParticleGrid()
{
    deleteBuffers();
}

void
ParticleGrid::setGrid( const Grid &grid )
{
    m_grid = grid;
    LOG( "Grid dimensions: %d %d %d", grid.dim.x, grid.dim.y, grid.dim.z );
    m_size = m_grid.nodeCount();
    LOG( "Node count: %d (should be %d)", m_size, (grid.dim.x+1)*(grid.dim.y+1)*(grid.dim.z+1) );
    deleteBuffers();
}

void
ParticleGrid::render()
{
    if ( size() > 0 ) {

        if ( !hasBuffers() ) buildBuffers();

        QGLShaderProgram *shader = ParticleGrid::shader();
        if ( shader ) {
            glPushAttrib( GL_VERTEX_PROGRAM_POINT_SIZE );
            glEnable( GL_VERTEX_PROGRAM_POINT_SIZE );
            shader->bind();
            shader->setUniformValue( "pos", m_grid.pos.x, m_grid.pos.y, m_grid.pos.z );
            shader->setUniformValue( "dim", (float)m_grid.dim.x, (float)m_grid.dim.y, (float)m_grid.dim.z );
            shader->setUniformValue( "h", m_grid.h );
            shader->setUniformValue( "mode", UiSettings::showGridDataMode() );
        } else {
            glPushAttrib( GL_LIGHTING_BIT );
            glDisable( GL_LIGHTING );
            glColor4f( 1.0f, 1.0f, 1.0f, 0.f );
            glPointSize( 1.f );
        }

        glPushAttrib( GL_COLOR_BUFFER_BIT );
        glEnable( GL_BLEND );
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

        glPushAttrib( GL_DEPTH_BUFFER_BIT );
        glDepthMask( false );
        glEnable( GL_ALPHA_TEST );
        glAlphaFunc( GL_GREATER, 0.05f );

        glBindVertexArray( m_glVAO );
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_glIndices );
        glDrawElements( GL_POINTS, m_size, GL_UNSIGNED_INT, (void*)(0) );
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
        glBindVertexArray( 0 );

        glPopAttrib();
        glPopAttrib();

        if ( shader ) {
            shader->release();
            glPopAttrib();
        } else {
            glPopAttrib();
        }

    }
}

bool
ParticleGrid::hasBuffers() const
{
    return m_glVBO > 0 && glIsBuffer( m_glVBO );
}

void
ParticleGrid::buildBuffers()
{
    deleteBuffers();

    ParticleGridNode *data = new ParticleGridNode[m_size];
    memset( data, 0, m_size*sizeof(ParticleGridNode) );

    // Build VBO
    glGenBuffers( 1, &m_glVBO );
    glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
    glBufferData( GL_ARRAY_BUFFER, m_size*sizeof(ParticleGridNode), data, GL_DYNAMIC_DRAW );

    delete [] data;

    // Build VAO
    glGenVertexArrays( 1, &m_glVAO );
    glBindVertexArray( m_glVAO );

    // Mass attribute
    glEnableVertexAttribArray( 0 );
    glVertexAttribPointer( 0, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleGridNode), (void*)(0) );

    // Velocity attribute
    glEnableVertexAttribArray( 1 );
    glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleGridNode), (void*)(sizeof(GLfloat)) );

    // Force attribute
    glEnableVertexAttribArray( 2 );
    glVertexAttribPointer( 2, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleGridNode), (void*)(sizeof(GLfloat)+2*sizeof(vec3)) );

    glBindVertexArray( 0 );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    // Indices (needed to access vertex index in shader)
    QVector<unsigned int> indices;
    for ( unsigned int i = 0; i < (unsigned int)m_size; ++i ) indices += i;

    glGenBuffers( 1, &m_glIndices );
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_glIndices );
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned int), indices.data(), GL_DYNAMIC_DRAW );
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );

}

void
ParticleGrid::deleteBuffers()
{
    // Delete OpenGL VBO and unregister with CUDA
    if ( hasBuffers() ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
        glDeleteBuffers( 1, &m_glVBO );
        glDeleteVertexArrays( 1, &m_glVAO );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_glIndices );
        glDeleteBuffers( 1, &m_glIndices );
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
    }
    m_glVBO = 0;
    m_glVAO = 0;
    m_glIndices = 0;
}

BBox
ParticleGrid::getBBox( const glm::mat4 &ctm )
{
    return BBox(m_grid.pos+m_grid.h*vec3(m_grid.dim.x,m_grid.dim.y,m_grid.dim.z)).getBBox(ctm);
}

QGLShaderProgram* ParticleGrid::SHADER = NULL;

QGLShaderProgram*
ParticleGrid::shader()
{
    if ( SHADER == NULL ) {
        const QGLContext *context = QGLContext::currentContext();
        SHADER = new QGLShaderProgram( context );
        if ( !SHADER->addShaderFromSourceFile(QGLShader::Vertex, ":/shaders/particlegrid.vert") ||
             !SHADER->addShaderFromSourceFile(QGLShader::Fragment, ":/shaders/particlegrid.frag") ) {
            LOG( "ParticleGrid::shader() : Compile error: \n%s\n", STR(SHADER->log().trimmed()) );
            SAFE_DELETE( SHADER );
        } else {
            SHADER->bindAttributeLocation( "nodeMass", 0 );
            SHADER->bindAttributeLocation( "nodeVelocity", 1 );
            SHADER->bindAttributeLocation( "nodeForce", 2 );
            glBindFragDataLocation( SHADER->programId(), 0, "fragmentColor" );
            if ( !SHADER->link() ) {
                LOG( "ParticleGrid::shader() : Link error: \n%s\n", STR(SHADER->log().trimmed()) );
                SAFE_DELETE( SHADER );
            }
        }
    }
    return SHADER;
}
