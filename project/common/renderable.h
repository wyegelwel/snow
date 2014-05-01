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
#include "glm/common.hpp"

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

    virtual void renderVelForPicker() {}

    // These functions are used by the SceneNodes to cache their renderable's
    // bounding box or centroid. The object computes its bounding box or
    // centroid in its local coordinate frame, and then transforms it to
    // the SceneNode's using the CTM;
    virtual BBox getBBox( const glm::mat4 &ctm = glm::mat4(1.f) ) = 0;
    virtual vec3 getCentroid( const glm::mat4 &ctm = glm::mat4(1.f) ) = 0;

    // Used for scene interaction.
    virtual void setSelected( bool selected ) { m_selected = selected; }
    bool isSelected() const { return m_selected; }

    virtual void rotateVelVec(const glm::mat4 &transform){glm::vec4 v(m_velVec.x,m_velVec.y,m_velVec.z,1);v=transform*v;m_velVec=glm::vec3(v.x,v.y,v.z);}

    virtual void setVelMag(const float m)  {m_velMag = m;}

    virtual void setVelVec(const glm::vec3 &vec){m_velVec = vec;}

    virtual void updateMeshVel(){}

    virtual float getVelMag() {return m_velMag;}

    virtual glm::vec3 getVelVec() {return m_velVec;}

    virtual void setCTM(const glm::mat4 &ctm){m_ctm = ctm;}

protected:

    bool m_selected;
    glm::vec3 m_velVec;
    float m_velMag;
    glm::mat4 m_ctm;
};

#endif // RENDERABLE_H
