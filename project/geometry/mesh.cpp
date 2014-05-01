/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mesh.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 8 Apr 2014
**
**************************************************************************/

#include "mesh.h"

#include <GL/gl.h>

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/geometric.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <QElapsedTimer>
#include <QLocale>

#include "common/common.h"
#include "cuda/functions.h"
#include "geometry/bbox.h"
#include "geometry/grid.h"
#include "sim/particlesystem.h"
#include "ui/uisettings.h"

Mesh::Mesh()
    : m_glVBO(0),
      m_color(0.4f, 0.4f, 0.4f, 1.f)
{
}

Mesh::Mesh( const QVector<Vertex> &vertices,
            const QVector<Tri> &tris )
    : m_vertices(vertices),
      m_tris(tris),
      m_glVBO(0),
      m_cudaVBO(NULL),
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
      m_glVBO(0),
      m_cudaVBO(NULL),
      m_color(0.5f, 0.5f, 0.5f, 1.f)
{
}

Mesh::Mesh( const Mesh &mesh )
    : m_name(mesh.m_name),
      m_filename(mesh.m_filename),
      m_vertices(mesh.m_vertices),
      m_tris(mesh.m_tris),
      m_normals(mesh.m_normals),
      m_glVBO(0),
      m_cudaVBO(NULL),
      m_color(mesh.m_color)
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
        Normal n = vec3::cross(v1-v0, v2-v0);
        triAreas[i] = vec3::length(n)/2.f;
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

    if ( ( m_type == SNOW_CONTAINER ) ? UiSettings::showContainers() : UiSettings::showColliders() ) {

        glPushAttrib( GL_DEPTH_TEST );
        glEnable( GL_DEPTH_TEST );

        glEnable( GL_LINE_SMOOTH );
        glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );

        glPushAttrib( GL_COLOR_BUFFER_BIT );
        glEnable( GL_BLEND );
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

        glm::vec4 color = ( m_selected ) ? glm::mix( m_color, UiSettings::selectionColor(), 0.5f ) : m_color;

        if ( (m_type == SNOW_CONTAINER) ?
             (UiSettings::showContainersMode() == UiSettings::SOLID || UiSettings::showContainersMode() == UiSettings::SOLID_AND_WIREFRAME ) :
             (UiSettings::showCollidersMode() == UiSettings::SOLID || UiSettings::showCollidersMode() == UiSettings::SOLID_AND_WIREFRAME ) ) {
            glPushAttrib( GL_LIGHTING_BIT );
            glEnable( GL_LIGHTING );
            glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, glm::value_ptr(color*0.2f) );
            glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, glm::value_ptr(color) );
            glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
            renderVBO();
            glPopAttrib();
        }

        if ( (m_type == SNOW_CONTAINER) ?
             (UiSettings::showContainersMode() == UiSettings::WIREFRAME || UiSettings::showContainersMode() == UiSettings::SOLID_AND_WIREFRAME) :
             (UiSettings::showCollidersMode() == UiSettings::WIREFRAME || UiSettings::showCollidersMode() == UiSettings::SOLID_AND_WIREFRAME ) ) {
            glPushAttrib( GL_POLYGON_BIT );
            glEnable( GL_POLYGON_OFFSET_LINE );
            glPolygonOffset( -1.f, -1.f );
            glPushAttrib( GL_LIGHTING_BIT );
            glDisable( GL_LIGHTING );
            glLineWidth( 1.f );
            glColor4fv( glm::value_ptr(color*0.8f) );
            glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
            renderVBO();
            glPopAttrib();
            glPopAttrib();
        }

        glPopAttrib();
        glPopAttrib();

    }
}

void
Mesh::renderForPicker()
{
    if ( !hasVBO() ) {
        buildVBO();
    }
    if ( (m_type == SNOW_CONTAINER) ? UiSettings::showContainers() : UiSettings::showColliders() ) {
        glPushAttrib( GL_DEPTH_TEST );
        glEnable( GL_DEPTH_TEST );
        glPushAttrib( GL_LIGHTING_BIT );
        glDisable( GL_LIGHTING );
        glColor3f( 1.f, 1.f, 1.f );
        renderVBO();
        glPopAttrib();
        glPopAttrib();
    }
}

void
Mesh::renderVBO()
{
    glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, 2*sizeof(vec3), (void*)(0) );
    glEnableClientState( GL_NORMAL_ARRAY );
    glNormalPointer( GL_FLOAT, 2*sizeof(vec3), (void*)(sizeof(vec3)) );

    glDrawArrays( GL_TRIANGLES, 0, 3*getNumTris() );

    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glDisableClientState( GL_VERTEX_ARRAY );
    glDisableClientState( GL_NORMAL_ARRAY );
}

bool
Mesh::hasVBO() const
{
    bool has = false;
    if ( m_glVBO > 0 ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
        has = glIsBuffer( m_glVBO );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
    }
    return has;
}

void
Mesh::deleteVBO()
{
    if ( m_glVBO > 0 ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
        if ( glIsBuffer(m_glVBO) ) {
            unregisterVBO( m_cudaVBO );
            glDeleteBuffers( 1, &m_glVBO );
        }
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        m_glVBO = 0;
    }
}

void
Mesh::buildVBO()
{
    deleteVBO();

    // Create flat array of non-indexed triangles
    vec3 *data = new vec3[6*getNumTris()];
    for ( int i = 0, index = 0; i < getNumTris(); ++i ) {
        const Tri &tri = m_tris[i];
        data[index++] = m_vertices[tri[0]];
        data[index++] = m_normals[tri[0]];
        data[index++] = m_vertices[tri[1]];
        data[index++] = m_normals[tri[1]];
        data[index++] = m_vertices[tri[2]];
        data[index++] = m_normals[tri[2]];
    }

    // Build OpenGL VBO
    glGenBuffers( 1, &m_glVBO );
    glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
    glBufferData( GL_ARRAY_BUFFER, 6*getNumTris()*sizeof(vec3), data, GL_STATIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    // Register with CUDA
    registerVBO( &m_cudaVBO, m_glVBO );

    delete [] data;
}

void
Mesh::fill( ParticleSystem &particles, int particleCount, float h, float targetDensity )
{
    if ( !hasVBO() ) {
        buildVBO();
    }

    QElapsedTimer timer;
    timer.start();

    Grid grid = getObjectBBox().toGrid( h );

    LOG( "Filling mesh in %d x %d x %d grid (%s voxels)...", grid.dim.x, grid.dim.y, grid.dim.z, STR(QLocale().toString(grid.dim.x*grid.dim.y*grid.dim.z)) );

    particles.resize( particleCount );
    fillMesh( &m_cudaVBO, getNumTris(), grid, particles.data(), particleCount, targetDensity,  m_fillMaterialPreset );

#if 0
    fillMesh2(&m_cudaVBO, getNumTris(), grid, particles.data(), particleCount, targetDensity);
#endif


    LOG( "Mesh filled with %s particles in %lld ms.", STR(QLocale().toString(particleCount)), timer.restart() );
}

BBox
Mesh::getBBox( const glm::mat4 &ctm )
{
    BBox box;
    for ( int i = 0; i < getNumVertices(); ++i ) {
        const Vertex &v = m_vertices[i];
        glm::vec4 point = ctm * glm::vec4( v.x, v.y, v.z, 1.f );
        box += vec3( point.x, point.y, point.z );
    }
    return box;
}

vec3
Mesh::getCentroid( const glm::mat4 &ctm )
{
    vec3 c(0,0,0);
    for ( int i = 0; i < getNumVertices(); ++i ) {
        const Vertex &v = m_vertices[i];
        glm::vec4 point = ctm * glm::vec4( v.x, v.y, v.z, 1.f );
        c += vec3( point.x, point.y, point.z );
    }
    return c / (float)getNumVertices();
}

BBox
Mesh::getObjectBBox() const
{
    BBox box;
    for ( int i = 0; i < getNumVertices(); ++i ) {
        box += m_vertices[i];
    }
    return box;
}

void
Mesh::applyTransformation( const glm::mat4 &transform )
{
    for ( int i = 0; i < getNumVertices(); ++i ) {
        const Vertex &v = m_vertices[i];
        glm::vec4 point = transform * glm::vec4( v.x, v.y, v.z, 1.f );
        m_vertices[i] = vec3( point.x, point.y, point.z );
    }
    computeNormals();
    deleteVBO();
}

void
Mesh::append( const Mesh &mesh )
{
    int offset = m_vertices.size();
    for ( int i = 0; i < mesh.m_tris.size(); ++i ) {
        Tri tri = mesh.m_tris[i];
        tri.offset( offset );
        m_tris += tri;
    }
    m_vertices += mesh.m_vertices;
    m_normals += mesh.m_normals;
    deleteVBO();
}
