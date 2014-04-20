/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   renderable.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef RENDERABLE_H
#define RENDERABLE_H

class Renderable
{

public:

    Renderable() : m_selected(false) {}
    virtual ~Renderable() {}
    virtual void render() {}
    virtual void renderForPicker() {}

    void setSelected( bool selected ) { m_selected = selected; }
    bool isSelected() const { return m_selected; }

    bool m_selected;
};

#endif // RENDERABLE_H
