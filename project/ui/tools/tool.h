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

};

#endif // TOOL_H
