#include "collider.h"
#include "io/objparser.h"
#include "qgl.h"
#include <iostream>

//#define PLANE_CONSTANT 50.0;

float const PLANE_CONSTANT = 10.0f;

ImplicitCollider::ImplicitCollider(ColliderType t, vec3 c, vec3 p)  {
    type = t;
    center = c;
    param = p;
    initializeMesh();
}

void ImplicitCollider::render()  {
    glPushMatrix();
    glTranslatef(center.x,center.y,center.z);
    switch(type)  {
    case(SPHERE):
        renderSphere();
        break;
    case(HALF_PLANE):
        renderPlane();
        break;
    default:
        break;
    }
    mesh->render();
    glPopMatrix();
}

void ImplicitCollider::renderSphere() {
    glScalef(param.x,param.x,param.x);
}

void ImplicitCollider::renderPlane()  {
    vec3 oldNormal = vec3(0,1,0);
    vec3 rotationAxis = vec3::cross(oldNormal,param);
    float rotationAngle = acos(vec3::dot(param,oldNormal));
    rotationAngle *= (180.0/M_PI);
    glRotatef(rotationAngle,rotationAxis.x,rotationAxis.y,rotationAxis.z);
    glScalef(PLANE_CONSTANT,PLANE_CONSTANT,PLANE_CONSTANT);
}

void ImplicitCollider::initializeMesh()  {
        QList<Mesh*> colliderMeshes;
        switch(type)  {
        case SPHERE:
            OBJParser::load( PROJECT_PATH "/data/models/sphere.obj", colliderMeshes);
            break;
        case HALF_PLANE:
            OBJParser::load( PROJECT_PATH "/data/models/plane.obj", colliderMeshes);
            break;
        default:
            break;
        }
        mesh = colliderMeshes[0];
}
