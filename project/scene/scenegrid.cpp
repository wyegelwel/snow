/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scenegrid.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 20 Apr 2014
**
**************************************************************************/

#include <GL/gl.h>

#include <QVector>

#include "scene/scenegrid.h"
#include "geometry/bbox.h"
#include "ui/uisettings.h"

SceneGrid::SceneGrid()
    : m_grid(),
      m_vbo(0),
      m_vboSize(0)
{

}

SceneGrid::SceneGrid( const Grid &grid )
    : m_grid(grid),
      m_vbo(0),
      m_vboSize(0)
{

}

SceneGrid::~SceneGrid()
{
    deleteVBO();
}

void
SceneGrid::render()
{
    if ( !hasVBO() ) {
        buildVBO();
    }

    if ( m_vboSize > 0 && UiSettings::showBBox() ) {

        glPushAttrib( GL_DEPTH_BUFFER_BIT );
        glEnable( GL_DEPTH_TEST );

        glPushAttrib( GL_COLOR_BUFFER_BIT );
        glEnable( GL_BLEND );
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

        glEnable( GL_LINE_SMOOTH );
        glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );

        glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
        glEnableClientState( GL_VERTEX_ARRAY );
        glVertexPointer( 3, GL_FLOAT, sizeof(vec3), (void*)(0) );

        glColor4f( 0.5f, 0.8f, 1.f, 0.5f );
        glLineWidth( 3.f );
        glDrawArrays( GL_LINES, 0, 24 );

        if ( UiSettings::showGrid() ) {
            glColor4f( 0.5f, 0.8f, 1.f, 0.25f );
            glLineWidth( 0.5f );
            glDrawArrays( GL_LINES, 24, m_vboSize-24 );
        }

        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        glDisableClientState( GL_VERTEX_ARRAY );

        glPopAttrib();
        glPopAttrib();
    }

}

bool
SceneGrid::hasVBO() const
{
    return m_vbo > 0 && glIsBuffer( m_vbo );
}

void
SceneGrid::buildVBO()
{
    deleteVBO();

    QVector<vec3> data;

    const glm::ivec3 &dim = m_grid.dim;
    const float &h = m_grid.h;
    vec3 min = m_grid.pos;
    vec3 max = m_grid.pos + h * vec3( dim.x, dim.y, dim.z );

    // Bounding box
    data += min;
    data += vec3( min.x, min.y, max.z );
    data += vec3( min.x, min.y, max.z );
    data += vec3( min.x, max.y, max.z );
    data += vec3( min.x, max.y, max.z );
    data += vec3( min.x, max.y, min.z );
    data += vec3( min.x, max.y, min.z );
    data += min;
    data += vec3( max.x, min.y, min.z );
    data += vec3( max.x, min.y, max.z );
    data += vec3( max.x, min.y, max.z );
    data += vec3( max.x, max.y, max.z );
    data += vec3( max.x, max.y, max.z );
    data += vec3( max.x, max.y, min.z );
    data += vec3( max.x, max.y, min.z );
    data += vec3( max.x, min.y, min.z );
    data += min;
    data += vec3( max.x, min.y, min.z );
    data += vec3( min.x, min.y, max.z );
    data += vec3( max.x, min.y, max.z );
    data += vec3( min.x, max.y, max.z );
    data += max;
    data += vec3( min.x, max.y, min.z );
    data += vec3( max.x, max.y, min.z );

    // yz faces
    for ( int i = 1; i < dim.y; ++i ) {
        float y = min.y + i*h;
        data += vec3( min.x, y, min.z );
        data += vec3( min.x, y, max.z );
        data += vec3( max.x, y, min.z );
        data += vec3( max.x, y, max.z );
    }
    for ( int i = 1; i < dim.z; ++i ) {
        float z = min.z + i*h;
        data += vec3( min.x, min.y, z );
        data += vec3( min.x, max.y, z );
        data += vec3( max.x, min.y, z );
        data += vec3( max.x, max.y, z );
    }

    // xy faces
    for ( int i = 1; i < dim.x; ++i ) {
        float x = min.x + i*h;
        data += vec3( x, min.y, min.z );
        data += vec3( x, max.y, min.z );
        data += vec3( x, min.y, max.z );
        data += vec3( x, max.y, max.z );
    }
    for ( int i = 1; i < dim.y; ++i ) {
        float y = min.y + i*h;
        data += vec3( min.x, y, min.z );
        data += vec3( max.x, y, min.z );
        data += vec3( min.x, y, max.z );
        data += vec3( max.x, y, max.z );
    }

    // xz faces
    for ( int i = 1; i < dim.x; ++i ) {
        float x = min.x + i*h;
        data += vec3( x, min.y, min.z );
        data += vec3( x, min.y, max.z );
        data += vec3( x, max.y, min.z );
        data += vec3( x, max.y, max.z );
    }
    for ( int i = 1; i < dim.z; ++i ) {
        float z = min.z + i*h;
        data += vec3( min.x, min.y, z );
        data += vec3( max.x, min.y, z );
        data += vec3( min.x, max.y, z );
        data += vec3( max.x, max.y, z );
    }

    m_vboSize = data.size();

    glGenBuffers( 1, &m_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glBufferData( GL_ARRAY_BUFFER, data.size()*sizeof(vec3), data.data(), GL_STATIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
}

void
SceneGrid::deleteVBO()
{
    if ( m_vbo > 0 ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
        if ( glIsBuffer(m_vbo) ) glDeleteBuffers( 1, &m_vbo );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        m_vbo = 0;
    }
}

BBox
SceneGrid::getBBox( const glm::mat4 &ctm )
{
    return (BBox(m_grid.pos,m_grid.pos+m_grid.h*vec3(m_grid.dim.x, m_grid.dim.y, m_grid.dim.z))).getBBox( ctm );
}
