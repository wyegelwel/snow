/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   caches.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 28 Apr 2014
**
**************************************************************************/

#ifndef CACHES_H
#define CACHES_H

#include "cuda/matrix.cu"

struct NodeCache
{
    // Data used by Conjugate Residual Method
    vec3 *r;
    vec3 *s;
    vec3 *p;
    vec3 *q;
    vec3 *v;
    vec3 *df;
    float *scratch;
    NodeCache()
    {
        r = s = p = q = v = df = NULL;
        scratch = NULL;
    }
};

struct ParticleCache
{
    // Data used during initial node computations
    mat3 sigma;
    vec3 particleGridPos;

    // Data used during implicit node velocity update
    mat3 Ap;
    mat3 FeHat;
    mat3 ReHat;
    mat3 SeHat;
    mat3 dF;
};

#endif // CACHES_H
