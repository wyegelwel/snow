/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   types.h
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef TYPES_H
#define TYPES_H

#include "cuda/vector.h"

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include <glm/vec4.hpp>

typedef vec3 Vertex;
typedef vec3 Normal;
typedef glm::vec4 Color;

#endif // TYPES_H
