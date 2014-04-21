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

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/mat4x4.hpp"

struct BBox;

class Renderable
{

public:

    Renderable() : m_selected(false) {}
    virtual ~Renderable() {}
    virtual void render() {}
    virtual void renderForPicker() {}

    virtual void setSelected( bool selected ) { m_selected = selected; }
    bool isSelected() const { return m_selected; }

    virtual BBox getBBox( const glm::mat4 &ctm = glm::mat4(1.f) ) = 0;

    bool m_selected;
};

#endif // RENDERABLE_H
