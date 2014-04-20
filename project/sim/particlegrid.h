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

#include "cuda/matrix.cu"
#include "cuda/vector.cu"


struct ParticleGrid : public Grid
{  
    struct Node
    {
        float mass;
        vec3 velocity;
        vec3 velocityChange; // v_n+1 - v_n (store this value through steps 4,5,6)
        vec3 force;

        Node() : mass(0), velocity(0,0,0), force(0,0,0) {}
    };
    inline Node* createNodes() const { return new Node[nodeCount()]; }
};

struct ParticleGridTempData{
    mat3 sigma;
    vec3 particleGridPos;
};


#endif // PARTICLEGRID_H
