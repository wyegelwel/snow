/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scenegrid.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 20 Apr 2014
**
**************************************************************************/

#ifndef SCENEGRID_H
#define SCENEGRID_H

#include "common/renderable.h"

#include "geometry/grid.h"

typedef unsigned int GLuint;

class SceneGrid : public Renderable
{

public:

    SceneGrid();
    SceneGrid( const Grid &grid );
    virtual ~SceneGrid();

    virtual void render();
    virtual void renderForPicker() { render(); }

    virtual BBox getBBox( const glm::mat4 &ctm );

    void setGrid( const Grid &grid ) { m_grid = grid; deleteVBO(); }

private:

    Grid m_grid;

    GLuint m_vbo;
    int m_vboSize;

    bool hasVBO() const;
    void buildVBO();
    void deleteVBO();

};

#endif // SCENEGRID_H
