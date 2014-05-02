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

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/matrix.h"

struct NodeCache
{
    enum Offset { R, AR, P, AP, V, DF };

    // Data used by Conjugate Residual Method
    vec3 r;
    vec3 Ar;
    vec3 p;
    vec3 Ap;
    vec3 v;
    vec3 df;
    double scratch;
    __host__ __device__ vec3& operator [] ( Offset i )
    {
        switch ( i ) {
        case R: return r;
        case AR: return Ar;
        case P: return p;
        case AP: return Ap;
        case V: return v;
        case DF: return df;
        }
        return r;
    }

    __host__ __device__ vec3 operator [] ( Offset i ) const
    {
        switch ( i ) {
        case R: return r;
        case AR: return Ar;
        case P: return p;
        case AP: return Ap;
        case V: return v;
        case DF: return df;
        }
        return r;
    }

};

struct ParticleCache
{
    // Data used during initial node computations
    mat3 *sigmas;

    // Data used during implicit node velocity update
    mat3 *Aps;
    mat3 *FeHats;
    mat3 *ReHats;
    mat3 *SeHats;
    mat3 *dFs;
};

#endif // CACHES_H
