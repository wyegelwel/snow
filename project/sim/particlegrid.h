/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   grid.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#ifndef PARTICLEGRID_H
#define PARTICLEGRID_H

#include "geometry/grid.h"

#ifdef CUDA_INCLUDE
    #include "cuda/vector.cu"
#else
    #include <glm/vec3.hpp>
#endif


struct ParticleGrid : public Grid
{  
    struct Node
    {
#ifdef CUDA_INCLUDE
    typedef vec3 vector_type;
#else
    typedef glm::vec3 vector_type;
#endif
        float mass;
        vector_type velocity;
        vector_type velocityChange; // v_n+1 - v_n (store this value through steps 4,5,6)
        vector_type force;

        Node() : mass(0), velocity(0,0,0), force(0,0,0) {}
    };

    inline Node* createNodes() const { return new Node[nodeCount()]; }
};

#endif // PARTICLEGRID_H
