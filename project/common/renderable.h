/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   renderable.h
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef RENDERABLE_H
#define RENDERABLE_H

class Renderable
{
public:
    Renderable() {}
    virtual ~Renderable() {}
    virtual void render() {}
    bool renderOnSim = true;
};

#endif // RENDERABLE_H
