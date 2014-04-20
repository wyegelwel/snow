/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   grid.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#ifndef GRID_H
#define GRID_H

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/vec3.hpp"

#include "cuda/vector.cu"

struct Grid
{
    glm::ivec3 dim;
    vec3 pos;
    float h;

    Grid() : dim(0,0,0), pos(0,0,0), h(0.f) {}

    inline bool empty() const { return cellCount() == 0; }

    inline glm::ivec3 nodeDim() const {return dim+1;}
    inline int nodeCount() const { return (dim.x+1)*(dim.y+1)*(dim.z+1); }
    inline int cellCount() const { return dim.x * dim.y * dim.z; }
    inline int index( int i, int j, int k ) const { return (i*(dim.y*dim.z) + j*(dim.z) + k); }

};

#endif // GRID_H
