#ifndef COLLIDER_H
#define COLLIDER_H

#include "cuda/vector.cu"

/**
 * For the sake of supporting multiple implicit colliders in cuda, we define an enum for the type of collider
 * and use the ImplicitCollider.param to specify the collider once the type is known. Most simple implicit shapes
 * can be paramterized using at most 3 parameters. For instance, a half-plane is a point (ImplicitCollider.center)
 * and a normal (ImplicitCollider.param). A sphere is a center (ImplicitCollider.center) and a radius (ImplicitCollider.param.x)
 */

enum ColliderType{
    HALF_PLANE, SPHERE
};

struct ImplicitCollider {
    vec3 center;
    vec3 param;
    vec3 velocity;
    ColliderType type;
};

#ifndef CUDA_INCLUDE

#include "common/renderable.h"
#include "geometry/mesh.h"

class Collider : public Renderable {

public:


    Collider( ImplicitCollider &collider );
    virtual void render();//{mesh->render();}

    virtual void init() = 0;

    void initializeMesh();

private:

    ImplicitCollider &m_collider;
    Mesh *m_mesh;

    void renderSphere();
    void renderPlane();

};

#endif //CUDA_INCLUDE

#endif // COLLIDER_H
