#include "collider.h"
#include "io/objparser.h"
#include "qgl.h"
#include <iostream>

//#define PLANE_CONSTANT 50.0;

float const PLANE_CONSTANT = 10.0f;

Collider::Collider( ImplicitCollider &collider )
    : m_collider(collider)
{
    initializeMesh();
}

void Collider::render()
{
    glPushMatrix();
    glTranslatef( m_collider.center.x, m_collider.center.y, m_collider.center.z );
    switch( m_collider.type )  {
    case(SPHERE):
        renderSphere();
        break;
    case(HALF_PLANE):
        renderPlane();
        break;
    default:
        break;
    }
    m_mesh->render();
    glPopMatrix();
}

void Collider::renderSphere()
{
    glScalef( m_collider.param.x, m_collider.param.x, m_collider.param.x );
}

void Collider::renderPlane()
{
    vec3 oldNormal = vec3(0,1,0);
    vec3 rotationAxis = vec3::cross( oldNormal, m_collider.param );
    float rotationAngle = acos(vec3::dot(m_collider.param,oldNormal));
    rotationAngle *= (180.0/M_PI);
    glRotatef(rotationAngle,rotationAxis.x,rotationAxis.y,rotationAxis.z);
    glScalef(PLANE_CONSTANT,PLANE_CONSTANT,PLANE_CONSTANT);
}

void Collider::initializeMesh()  {
        QList<Mesh*> colliderMeshes;
        switch( m_collider.type )  {
        case SPHERE:
            OBJParser::load( PROJECT_PATH "/data/models/sphere.obj", colliderMeshes);
            break;
        case HALF_PLANE:
            OBJParser::load( PROJECT_PATH "/data/models/plane.obj", colliderMeshes);
            break;
        default:
            break;
        }
        m_mesh = colliderMeshes[0];
}
