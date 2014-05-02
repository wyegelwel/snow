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
struct ParticleCache;
struct Node;
struct NodeCache;
struct ImplicitCollider;
struct SimulationParameters;
struct Material;

extern "C"
{

// OpenGL-CUDA interop
void registerVBO( cudaGraphicsResource **resource, GLuint vbo );
void unregisterVBO( cudaGraphicsResource *resource );

// Particle simulation
void updateParticles( Particle *particles, ParticleCache *devParticleCache, ParticleCache *hostParticleCache, int numParticles,
                      Grid *grid, Node *nodes, NodeCache *nodeCache, int numNodes,
                      ImplicitCollider *colliders, int numColliders,
                      float timeStep, bool implicitUpdate );

// Mesh filling
void fillMesh( cudaGraphicsResource **resource, int triCount, const Grid &grid, Particle *particles, int particleCount, float targetDensity, int materialPreset);

#if 0
void fillMesh2( cudaGraphicsResource **resource, int triCount, const Grid &grid, Particle *particles, int particleCount, float targetDensity);
#endif

// One time computation to get particle volumes
void initializeParticleVolumes( Particle *particles, int numParticles, const Grid *grid, int numNodes );

}

#endif // FUNCTIONS_H
