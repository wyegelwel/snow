#ifndef COLLIDER_H
#define COLLIDER_H

#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

/**
 * For the sake of supporting multiple implicit colliders in cuda, we define
 *
 */

enum ColliderType{
    HALF_PLANE
};

struct ImplicitCollider{
    glm::vec3 center;
    glm::vec3 param;
    ColliderType type;
};

#endif // COLLIDER_H
