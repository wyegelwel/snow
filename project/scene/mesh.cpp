/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mesh.cpp
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#include "mesh.h"

#include <GL/gl.h>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "common/common.h"

Mesh::Mesh()
    : m_vbo(0),
      m_color(0.5f, 0.5f, 0.5f, 1.f)
{
}

Mesh::Mesh( const QVector<Vertex> &vertices,
            const QVector<Tri> &tris )
    : m_vertices(vertices),
      m_tris(tris),
      m_vbo(0),
      m_color(0.5f, 0.5f, 0.5f, 1.f)
{
    computeNormals();
}

Mesh::Mesh( const QVector<Vertex> &vertices,
            const QVector<Tri> &tris,
            const QVector<Normal> &normals )
    : m_vertices(vertices),
      m_tris(tris),
      m_normals(normals),
      m_vbo(0),
      m_color(0.5f, 0.5f, 0.5f, 1.f)
{
}

Mesh::~Mesh()
{
    deleteVBO();
}

void
Mesh::computeNormals()
{
    Normal *triNormals = new Normal[getNumTris()];
    float *triAreas = new float[getNumTris()];
    QVector<int> *vertexMembership = new QVector<int>[getNumVertices()];
    for ( int i = 0; i < getNumTris(); ++i ) {
        // Compute triangle normal and area
        const Tri &tri = m_tris[i];
        const Vertex &v0 = m_vertices[tri[0]];
        const Vertex &v1 = m_vertices[tri[1]];
        const Vertex &v2 = m_vertices[tri[2]];
        Normal n = glm::cross(v1-v0, v2-v0);
        triAreas[i] = glm::length(n)/2.f;
        triNormals[i] = 2.f*n/triAreas[i];
        // Record triangle membership for each vertex
        vertexMembership[tri[0]] += i;
        vertexMembership[tri[1]] += i;
        vertexMembership[tri[2]] += i;
    }

    m_normals.clear();
    m_normals.resize( getNumVertices() );
    for ( int i = 0; i < getNumVertices(); ++i ) {
        Normal normal = Normal( 0.f, 0.f, 0.f );
        float sum = 0.f;
        for ( int j = 0; j < vertexMembership[i].size(); ++j ) {
            int index = vertexMembership[i][j];
            normal += triAreas[index]*triNormals[index];
            sum += triAreas[index];
        }
        normal /= sum;
        m_normals[i] = normal;
    }

    delete [] triNormals;
    delete [] triAreas;
    delete [] vertexMembership;
}

void
Mesh::render()
{
    if ( !hasVBO() ) {
        buildVBO();
    }

    glPushAttrib( GL_DEPTH_TEST );
    glEnable( GL_DEPTH_TEST );

    glPushAttrib( GL_POLYGON_BIT );
    glEnable( GL_POLYGON_OFFSET_LINE );
    glPolygonOffset( -1.f, -1.f );

    glColor4fv( glm::value_ptr(m_color) );
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    renderVBO();

    glLineWidth( 1.f );
    glColor4fv( glm::value_ptr(m_color*0.8f) );
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    renderVBO();

    glPopAttrib();
    glPopAttrib();
}

void
Mesh::renderVBO()
{
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, 2*sizeof(glm::vec3), (void*)(0) );
    glEnableClientState( GL_NORMAL_ARRAY );
    glNormalPointer( GL_FLOAT, 2*sizeof(glm::vec3), (void*)(sizeof(glm::vec3)) );

    glDrawArrays( GL_TRIANGLES, 0, 3*getNumTris() );

    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glDisableClientState( GL_VERTEX_ARRAY );
    glDisableClientState( GL_NORMAL_ARRAY );
}

bool
Mesh::hasVBO() const
{
    bool has = false;
    if ( m_vbo > 0 ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
        has = glIsBuffer( m_vbo );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
    }
    return has;
}

void
Mesh::deleteVBO()
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

void
Mesh::buildVBO()
{
    deleteVBO();

    glm::vec3 *data = new glm::vec3[6*getNumTris()];
    for ( int i = 0, index = 0; i < getNumTris(); ++i ) {
        const Tri &tri = m_tris[i];
        data[index++] = m_vertices[tri[0]];
        data[index++] = m_normals[tri[0]];
        data[index++] = m_vertices[tri[1]];
        data[index++] = m_normals[tri[1]];
        data[index++] = m_vertices[tri[2]];
        data[index++] = m_normals[tri[2]];
    }

    glGenBuffers( 1, &m_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glBufferData( GL_ARRAY_BUFFER, 6*getNumTris()*sizeof(glm::vec3), data, GL_STATIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    delete [] data;
}


