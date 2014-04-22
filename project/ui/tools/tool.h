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

class Tool
{

public:

    enum Type
    {
        SELECTION,
        MOVE,
        ROTATE,
        SCALE
    };

    Tool( ViewPanel *panel ) : m_panel(panel), m_mouseDown(false) {}
    virtual ~Tool() {}

    virtual void mousePressed() { m_mouseDown = true; }
    virtual void mouseMoved() {}
    virtual void mouseReleased() { m_mouseDown = false; }

    virtual void update() {}

    virtual void render() {}

protected:

    ViewPanel *m_panel;
    bool m_mouseDown;

    static glm::mat4 axialBasis( unsigned int axis )
    {
        const float m[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        unsigned int x = (axis+2)%3;
        unsigned int y = axis;
        unsigned int z = (axis+1)%3;
        return glm::mat4( m[x], m[3+x], m[6+x], 0,
                          m[y], m[3+y], m[6+y], 0,
                          m[z], m[3+z], m[6+z], 0,
                             0,      0,      0, 1 );
    }

};

#endif // TOOL_H
