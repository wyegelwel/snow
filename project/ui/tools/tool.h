/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   tool.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 20 Apr 2014
**
**************************************************************************/

#ifndef TOOL_H
#define TOOL_H

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/mat4x4.hpp"

class ViewPanel;
struct vec3;

class Tool
{

public:

    enum Type
    {
        SELECTION,
        MOVE,
        ROTATE,
        SCALE,
        VELOCITY
    };

    Tool( ViewPanel *panel ) : m_panel(panel), m_mouseDown(false) {}
    virtual ~Tool() {}

    virtual void mousePressed() { m_mouseDown = true; }
    virtual void mouseMoved() {}
    virtual void mouseReleased() { m_mouseDown = false; }

    virtual void update() {}

    virtual void render() {}

    static vec3 getAxialColor( unsigned int axis );

protected:

    ViewPanel *m_panel;
    bool m_mouseDown;

    static glm::mat4 getAxialBasis( unsigned int axis );

    float getHandleSize( const vec3 &center ) const;

};

#endif // TOOL_H
