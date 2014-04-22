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

struct vec3;
struct BBox;

class Renderable
{

public:

    Renderable() : m_selected(false) {}
    virtual ~Renderable() {}

    virtual void render() {}

    // Skip fancy rendering and just put the primitives
    // onto the framebuffer for pick testing.
    virtual void renderForPicker() {}

    // These functions are used by the SceneNodes to cache their renderable's
    // bounding box or centroid. The object computes its bounding box or
    // centroid in its local coordinate frame, and then transforms it to
    // the SceneNode's using the CTM;
    virtual BBox getBBox( const glm::mat4 &ctm = glm::mat4(1.f) ) = 0;
    virtual vec3 getCentroid( const glm::mat4 &ctm = glm::mat4(1.f) ) = 0;

    // Used for scene interaction.
    virtual void setSelected( bool selected ) { m_selected = selected; }
    bool isSelected() const { return m_selected; }

protected:

    bool m_selected;
};

#endif // RENDERABLE_H
