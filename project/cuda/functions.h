/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   functions.h
**   Author: mliberma
**   Created: 7 Apr 2014
**
**************************************************************************/

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

struct BBox;
struct Particle;

extern "C"
{

// OpenGL-CUDA interop
void registerVBO( cudaGraphicsResource **resource, GLuint vbo );
void unregisterVBO( cudaGraphicsResource *resource );

// Particle simulation
void updateParticles( cudaGraphicsResource **resource, float time, int particleCount );

// Mesh filling
void fillMesh( cudaGraphicsResource **resource, int triCount, const BBox &box, float h, Particle *particles, int particleCount );

}

#endif // FUNCTIONS_H
