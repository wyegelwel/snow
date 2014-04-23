/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   rotatetool.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 21 Apr 2014
**
**************************************************************************/

#ifndef ROTATETOOL_H
#define ROTATETOOL_H

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/vec2.hpp"

#include "ui/tools/selectiontool.h"

#include "cuda/vector.cu"

typedef unsigned int GLuint;

class RotateTool : public SelectionTool
{

public:

    RotateTool( ViewPanel *panel );
    virtual ~RotateTool();

    virtual void mousePressed();
    virtual void mouseMoved();
    virtual void mouseReleased();

    virtual void update();

    virtual void render();

protected:

    unsigned int m_axisSelection;

    bool m_active;
    bool m_rotating;
    vec3 m_center;
    float m_scale;

    GLuint m_vbo;
    int m_vboSize;

    void renderAxis( unsigned int i ) const;
    unsigned int getAxisPick() const;

    float intersectAxis( const glm::ivec2 &mouse ) const;

    bool hasVBO() const;
    void buildVBO();
    void deleteVBO();

};

#endif // ROTATETOOL_H
