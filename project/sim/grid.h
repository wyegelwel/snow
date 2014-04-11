#ifndef GRID_H
#define GRID_H

#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

struct Grid {
    glm::ivec3 dim;
    float h;
    glm::vec3 corner; // The world position of node (0,0,0)
};

#endif // GRID_H
