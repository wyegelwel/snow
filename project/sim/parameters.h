/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   parameters.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 19 Apr 2014
**
**************************************************************************/

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "cuda/vector.h"

struct SimulationParameters
{
    float timeStep;
    float startTime;
    float endTime;
    vec3 gravity;

    SimulationParameters()
    {
        timeStep = 0.001f;
        startTime = 0.f;
        endTime = 0.f;
        gravity = vec3( 0.f, -9.8f, 0.f );
    }
};

#endif // PARAMETERS_H
