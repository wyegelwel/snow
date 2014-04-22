/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scaletool.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 22 Apr 2014
**
**************************************************************************/

#ifndef SCALETOOL_H
#define SCALETOOL_H

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/vec2.hpp"

#include "ui/tools/selectiontool.h"

#include "cuda/vector.cu"

typedef unsigned int GLuint;

class ScaleTool : public SelectionTool
{

public:

    ScaleTool( ViewPanel *panel );
    virtual ~ScaleTool();

    virtual void mousePressed();
    virtual void mouseMoved();
    virtual void mouseReleased();

    virtual void update();

    virtual void render();

protected:

    unsigned int m_axisSelection;

    bool m_active;
    bool m_scaling;
    vec3 m_center;
    float m_scale;

    glm::ivec2 m_mouseDownPos;
    glm::mat4 m_transformInverse;
    glm::mat4 m_transform;

    GLuint  m_vbo;
    int m_vboSize;
    float m_radius;

    void renderAxis( unsigned int i ) const;
    void renderCenter() const;

    unsigned int getAxisPick() const;
    float intersectAxis( const glm::ivec2 &mouse ) const;

    bool hasVBO() const;
    void buildVBO();
    void deleteVBO();

};

#endif // SCALETOOL_H
