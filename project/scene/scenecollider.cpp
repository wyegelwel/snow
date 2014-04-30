/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scenecollider.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 29 Apr 2014
**
**************************************************************************/

#include "common/common.h"
#include "geometry/bbox.h"
#include "scene/scenecollider.h"
#include "sim/implicitcollider.h"
#include "io/objparser.h"

SceneCollider::SceneCollider( ImplicitCollider *collider )
    : m_collider(collider)
{
    initializeMesh();
}

SceneCollider::~SceneCollider()
{
    SAFE_DELETE( m_collider );
}

void SceneCollider::render()
{
    m_mesh->render();
}

void SceneCollider::renderForPicker()
{
    m_mesh->renderForPicker();
}

BBox
SceneCollider::getBBox( const glm::mat4 &ctm )
{
    return m_mesh->getBBox( ctm );
}

vec3
SceneCollider::getCentroid( const glm::mat4 &ctm )
{
    return m_mesh->getCentroid( ctm );
}

void
SceneCollider::setSelected( bool selected )
{
    m_selected = selected;
    m_mesh->setSelected( selected );
}

void SceneCollider::initializeMesh()
{
    QList<Mesh*> colliderMeshes;
    switch( m_collider->type ) {
    case SPHERE:
        OBJParser::load( PROJECT_PATH "/data/models/sphereCol.obj", colliderMeshes );
        break;
    case HALF_PLANE:
        OBJParser::load( PROJECT_PATH "/data/models/plane.obj", colliderMeshes );
        break;
    default:
        break;
    }
    m_mesh = colliderMeshes[0];
    m_mesh->setType( Mesh::COLLIDER );
}
