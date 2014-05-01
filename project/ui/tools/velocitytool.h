/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   velocitytool.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 21 Apr 2014
**
**************************************************************************/

#ifndef VELOCITYTOOL_H
#define VELOCITYTOOL_H

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/vec2.hpp"

#include "ui/tools/selectiontool.h"

#include "cuda/vector.h"

typedef unsigned int GLuint;

class VelocityTool : public SelectionTool
{

public:

    VelocityTool( ViewPanel *panel,Type t );
    virtual ~VelocityTool();

    virtual void mousePressed();
    virtual void mouseMoved();
    virtual void mouseReleased();

    virtual void update();

    virtual void render();

protected:

    unsigned int m_axisSelection,m_vecSelection;

    bool m_active;
    bool m_rotating,m_scaling;
    vec3 m_center;
    float m_scale;

    GLuint m_vbo;
    int m_vboSize;

    void renderAxis( unsigned int i ) const;
    unsigned int getAxisPick() const;
    unsigned int getVelVecPick() const;

    float intersectVelVec(const glm::ivec2 &mouse, const glm::vec3 &velVec) const;
    float intersectAxis( const glm::ivec2 &mouse ) const;

    bool hasVBO() const;
    void buildVBO();
    void deleteVBO();

};

#endif // VELOCITYTOOL_H
