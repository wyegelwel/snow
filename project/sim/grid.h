#ifndef GRID_H
#define GRID_H

#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

struct Grid {
    glm::ivec3 dim;
    glm::ivec3 pos;
    float h;
    int index(int i,int j,int k) {return (i*(dim.y*dim.z) + j*(dim.z) + k);}
};

#endif // GRID_H
