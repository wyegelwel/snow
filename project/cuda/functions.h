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

extern "C"
{

void registerVBO( cudaGraphicsResource **resource, GLuint vbo );
void unregisterVBO( cudaGraphicsResource *resource );

void updateParticles( cudaGraphicsResource **resource, float time, int particleCount );

}

#endif // FUNCTIONS_H
