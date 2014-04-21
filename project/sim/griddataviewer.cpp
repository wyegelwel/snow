/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   griddataviewer.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 20 Apr 2014
**
**************************************************************************/

#include <GL/gl.h>

#include <QVector>

#include "griddataviewer.h"

#include "common/common.h"
#include "common/math.h"
#include "geometry/bbox.h"
#include "sim/particlegridnode.h"
#include "ui/uisettings.h"

GridDataViewer::GridDataViewer( const Grid &grid )
    : m_grid(grid)
{
    m_vboSize = m_grid.nodeCount();
    m_data = new ParticleGridNode[m_vboSize];
    m_bytes = m_grid.nodeCount()*sizeof(ParticleGridNode);
    memset( m_data, 0, m_bytes );

    m_nodeVolume = m_grid.h * m_grid.h * m_grid.h;
}

GridDataViewer::~GridDataViewer()
{
    SAFE_DELETE_ARRAY( m_data );
    deleteVBO();
}

void
GridDataViewer::render()
{
    if ( !hasVBO() ) buildVBO();

    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, 7*sizeof(GLfloat), (void*)(0) );
    glEnableClientState( GL_COLOR_ARRAY );
    glColorPointer( 4, GL_FLOAT, 7*sizeof(GLfloat), (void*)(3*sizeof(GLfloat)) );

    glPushAttrib( GL_COLOR_BUFFER_BIT );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    glPushAttrib( GL_DEPTH_BUFFER_BIT );
    glDepthMask( false );
    glEnable( GL_ALPHA_TEST );
    glAlphaFunc( GL_GREATER, 0.05f );

    glPointSize( 2.f );
    glDrawArrays( GL_POINTS, 0, m_vboSize );

    glPopAttrib();
    glPopAttrib();

    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glDisableClientState( GL_VERTEX_ARRAY );
    glDisableClientState( GL_COLOR_ARRAY );

}

bool
GridDataViewer::hasVBO() const
{
    return m_vbo > 0 && glIsBuffer( m_vbo );
}

void
GridDataViewer::buildVBO()
{
    deleteVBO();

    void (GridDataViewer::*colorize)( const ParticleGridNode&, float&, float&, float&, float& ) const;
    switch ( (UiSettings::GridDataMode)(UiSettings::showGridDataMode()) ) {
    case UiSettings::NODE_MASS:
        colorize = &GridDataViewer::colorizeWithMass;
        break;
    case UiSettings::NODE_VELOCITY:
        colorize = &GridDataViewer::colorizeWithVelocity;
        break;
    default:
        colorize = &GridDataViewer::colorizeWithMass;
        break;
    }

    QVector<GLfloat> data;
    float r, g, b, a;
    for ( int i = 0, index = 0; i <= m_grid.dim.x; ++i ) {
        float x = m_grid.pos.x + i*m_grid.h;
        for ( int j = 0; j <= m_grid.dim.y; ++j ) {
            float y = m_grid.pos.y + j*m_grid.h;
            for ( int k = 0; k <= m_grid.dim.z; ++k, ++index ) {
                float z = m_grid.pos.z + k*m_grid.h;
                const ParticleGridNode &node = m_data[index];
                data += x; data += y; data += z; // Position
                (this->*colorize)( node, r, g, b, a );
                data += r; data += g; data += b; data += a; // Color
            }
        }
    }

    glGenBuffers( 1, &m_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glBufferData( GL_ARRAY_BUFFER, data.size()*sizeof(GLfloat), data.data(), GL_DYNAMIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
}

void
GridDataViewer::deleteVBO()
{
    if ( m_vbo > 0 ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
        if ( glIsBuffer(m_vbo) ) {
            glDeleteBuffers( 1, &m_vbo );
        }
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        m_vbo = 0;
    }
}

BBox
GridDataViewer::getBBox( const glm::mat4 &ctm )
{
    return BBox(m_grid.pos+m_grid.h*vec3(m_grid.dim.x,m_grid.dim.y,m_grid.dim.z)).getBBox(ctm);
}

void
GridDataViewer::colorizeWithMass( const ParticleGridNode &node, float &r, float &g, float &b, float &a ) const
{
    r = g = b = a = smoothstep( node.mass/m_nodeVolume, 0.f, 1e-3 );
}

void
GridDataViewer::colorizeWithVelocity( const ParticleGridNode &node, float &r, float &g, float &b, float &a ) const
{
    a = smoothstep( node.mass/m_nodeVolume, 0.f, 1e-3 );
    r = g = smoothstep( vec3::length(node.velocity), 0.f, 10.f );
    b = 1.f;
}
