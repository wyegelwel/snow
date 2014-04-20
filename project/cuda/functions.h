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

#include "sim/particlegridnode.h"

typedef unsigned int GLuint;
struct cudaGraphicsResource;
struct Grid;
struct Particle;
struct ParticleGridNode;
struct ParticleTempData;
struct ImplicitCollider;
struct MaterialConstants;
struct SimulationParameters;

extern "C"
{

// OpenGL-CUDA interop
void registerVBO( cudaGraphicsResource **resource, GLuint vbo );
void unregisterVBO( cudaGraphicsResource *resource );

// Particle simulation
void updateParticles( const SimulationParameters &parameters,
                      Particle *particles, int numParticles,
                      Grid *grid, ParticleGridNode *nodes, int numNodes, ParticleTempData *particleGridTempData,
                      ImplicitCollider *colliders, int numColliders,
                      MaterialConstants *mat );

// Mesh filling
void fillMesh( cudaGraphicsResource **resource, int triCount, const Grid &grid, Particle *particles, int particleCount );

}

#endif // FUNCTIONS_H
