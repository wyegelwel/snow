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

struct vec3;
struct mat3;

namespace Implicit {

// Cache for data used by Conjugate Residual Method
struct CRCache
{
    vec3 *r;
    vec3 *s;
    vec3 *p;
    vec3 *q;
    vec3 *u;
};

struct ACache
{
    mat3 Ap;
    mat3 FeHat;
    mat3 ReHat;
    mat3 SeHat;
    mat3 dF;
};

}

#endif // CACHES_H
