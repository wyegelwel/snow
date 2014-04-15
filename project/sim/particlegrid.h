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

#include "glm/vec3.hpp"

struct ParticleGrid : public Grid
{
    struct Node
    {
        float mass;
        glm::vec3 velocity;
        glm::vec3 velocityChange; // v_n+1 - v_n (store this value through steps 4,5,6)
        glm::vec3 force;

        Node() : mass(0), velocity(0,0,0), force(0,0,0) {}
    };

    inline Node* createNodes() const { return new Node[nodeCount()]; }
};

#endif // PARTICLEGRID_H
