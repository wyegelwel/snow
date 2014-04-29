/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   particlegridnode.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#ifndef PARTICLEGRIDNODE_H
#define PARTICLEGRIDNODE_H

#include "geometry/grid.h"

#include "cuda/matrix.cu"
#include "cuda/vector.cu"

struct Node
{  
    float mass;
    vec3 velocity;
    vec3 velocityChange; // v_n+1 - v_n (store this value through steps 4,5,6)
    vec3 force;
    Node() : mass(0), velocity(0,0,0), force(0,0,0) {}
};

#endif // PARTICLEGRIDNODE_H
