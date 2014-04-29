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

#ifndef FUNC
    #ifdef CUDA_INCLUDE
        #define FUNC __device__ __host__ __forceinline__
    #else
        #define FUNC inline
    #endif
#endif

#include "cuda/vector.h"

struct Grid
{
    glm::ivec3 dim;
    vec3 pos;
    float h;

    FUNC Grid() : dim(0,0,0), pos(0,0,0), h(0.f) {}
    FUNC Grid( const Grid &grid ) : dim(grid.dim), pos(grid.pos), h(grid.h) {}

    FUNC int cellCount() const { return dim.x * dim.y * dim.z; }
    FUNC bool empty() const { return dim.x*dim.y*dim.z == 0; }
    FUNC glm::ivec3 nodeDim() const { return dim + glm::ivec3(1,1,1); }

    FUNC int nodeCount() const { return (dim.x+1)*(dim.y+1)*(dim.z+1); }
    FUNC int index( int i, int j, int k ) const { return (i*(dim.y*dim.z) + j*(dim.z) + k); }

#define INDEX2IJK( I, J, K, INDEX, NI, NJ, NK )     \
{                                                   \
    I = INDEX / (NJ*NK);                            \
    INDEX = INDEX % (NJ*NK);                        \
    J = INDEX / NK;                                 \
    K = INDEX % NK;                                 \
}

    FUNC static void gridIndexToIJK( int idx, int &i, int &j, int &k, const glm::ivec3 &nodeDim )
    {
        INDEX2IJK( i, j, k, idx, nodeDim.x, nodeDim.y, nodeDim.z );
    }

    FUNC static void gridIndexToIJK( int idx, const glm::ivec3 &nodeDim, glm::ivec3 &ijk )
    {
        INDEX2IJK( ijk.x, ijk.y, ijk.z, idx, nodeDim.x, nodeDim.y, nodeDim.z );
    }

#undef INDEX2IJK

    FUNC static int getGridIndex( int i, int j, int k, const glm::ivec3 &nodeDim )
    {
        return (i*(nodeDim.y*nodeDim.z)+j*(nodeDim.z)+k);
    }

    FUNC static int getGridIndex( const glm::ivec3 &ijk, const glm::ivec3 &nodeDim )
    {
        return (ijk.x*(nodeDim.y*nodeDim.z)+ijk.y*(nodeDim.z)+ijk.z);
    }

    FUNC static bool withinBoundsInclusive( const float &v, const float &min, const float &max )
    {
        return ( v >= min && v <= max );
    }

    FUNC static bool withinBoundsInclusive( const glm::ivec3 &v, const glm::ivec3 &min, const glm::ivec3 &max )
    {
        return withinBoundsInclusive(v.x, min.x, max.x) && withinBoundsInclusive(v.y, min.y, max.y) && withinBoundsInclusive(v.z, min.z, max.z);
    }

};

#endif // GRID_H
