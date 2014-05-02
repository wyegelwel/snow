#ifndef COLLIDER_H
#define COLLIDER_H

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/mat4x4.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "cuda/vector.h"

/**
 * For the sake of supporting multiple implicit colliders in cuda, we define an enum for the type of collider
 * and use the ImplicitCollider.param to specify the collider once the type is known. Most simple implicit shapes
 * can be paramterized using at most 3 parameters. For instance, a half-plane is a point (ImplicitCollider.center)
 * and a normal (ImplicitCollider.param). A sphere is a center (ImplicitCollider.center) and a radius (ImplicitCollider.param.x)
 */

enum ColliderType
{
    HALF_PLANE = 0,
    SPHERE = 1
};

struct ImplicitCollider
{
    ColliderType type;
    vec3 center;
    vec3 param;
    vec3 velocity;
    float coeffFriction;

    __host__ __device__
    ImplicitCollider()
        : type(HALF_PLANE),
          center(0,0,0),
          param(0,1,0),
          velocity(0,0,0),
          coeffFriction(0.2f)
    {
    }

    __host__ __device__
    ImplicitCollider( ColliderType t, vec3 c, vec3 p = vec3(0,0,0), vec3 v = vec3(0,0,0), float f = 0.2f )
        : type(t),
          center(c),
          param(p),
          velocity(v),
          coeffFriction(f)
    {
        if ( p == vec3(0,0,0) ) {
            if ( t == HALF_PLANE ) p = vec3(0,1,0);
            else if ( t == SPHERE ) p = vec3(0.5f,0,0);
        }
    }

    __host__ __device__
    ImplicitCollider( const ImplicitCollider &collider )
        : type(collider.type),
          center(collider.center),
          param(collider.param),
          velocity(collider.velocity),
          coeffFriction(collider.coeffFriction)
    {
    }

    __host__ __device__
    void applyTransformation( const glm::mat4 &ctm )
    {
        glm::vec4 c = ctm * glm::vec4( glm::vec3(center), 1.f );
        center = vec3( c.x, c.y, c.z );
        switch ( type ) {
        case HALF_PLANE:
        {
            glm::vec4 n = ctm * glm::vec4( glm::vec3(param), 0.f );
            param = vec3( n.x, n.y, n.z );
            break;
        }
        case SPHERE:
        {
            const float *m = glm::value_ptr(ctm);
            param.x = sqrtf( m[0]*m[0] + m[1]*m[1] + m[2]*m[2] ); // Assumes uniform scale
            break;
        }
        }
    }

};

#endif // COLLIDER_H
