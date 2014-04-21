/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   particlegrid.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 21 Apr 2014
**
**************************************************************************/

#ifndef PARTICLEGRID_H
#define PARTICLEGRID_H

#include "common/renderable.h"
#include "geometry/grid.h"
#include "sim/particle.h"
#include "sim/particlegridnode.h"

class QGLShaderProgram;
typedef unsigned int GLuint;

class ParticleGrid : public Renderable
{

public:

    ParticleGrid();
    virtual ~ParticleGrid();

    virtual void render();

    void setGrid( const Grid &grid );
    Grid getGrid() const { return m_grid; }

    GLuint vbo() { if ( !hasBuffers() ) buildBuffers(); return m_glVBO; }

    inline int size() const { return m_size; }
    inline int nodeCount() const { return m_size; }

    virtual BBox getBBox( const glm::mat4 &ctm );

    bool hasBuffers() const;
    void buildBuffers();
    void deleteBuffers();

protected:

    static QGLShaderProgram *SHADER;
    static QGLShaderProgram* shader();

    Grid m_grid;
    int m_size;
    GLuint m_glIndices;
    GLuint m_glVBO;
    GLuint m_glVAO;

};

#endif // PARTICLEGRID_H
