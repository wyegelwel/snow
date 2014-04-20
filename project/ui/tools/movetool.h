/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   movetool.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 20 Apr 2014
**
**************************************************************************/

#ifndef MOVETOOL_H
#define MOVETOOL_H

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/vec2.hpp"

#include "ui/tools/selectiontool.h"

#include "cuda/vector.cu"

#include <QList>

class MoveTool : public SelectionTool
{

public:

    MoveTool( ViewPanel *panel );
    virtual ~MoveTool();

    virtual void mousePressed();
    virtual void mouseMoved();
    virtual void mouseReleased();

    virtual void update();

    virtual void render();

protected:

    unsigned int m_axisSelection;

    bool m_active;
    bool m_moving;
    vec3 m_center;

    void renderAxis( const vec3 &axis ) const;
    unsigned int getAxisPick() const;
    vec3 intersectAxis( const glm::ivec2 &mouse ) const;

};

#endif // MOVETOOL_H
