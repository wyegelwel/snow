/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   griddataviewer.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 20 Apr 2014
**
**************************************************************************/

#ifndef GRIDDATAVIEWER_H
#define GRIDDATAVIEWER_H

#include "common/renderable.h"
#include "geometry/grid.h"

struct ParticleGridNode;
typedef unsigned int GLuint;

class GridDataViewer : public Renderable
{

public:

    GridDataViewer( const Grid &grid );
    ~GridDataViewer();

    virtual void render();

    ParticleGridNode* data() { return m_data; }

    void update() { deleteVBO(); }

    int byteCount() const { return m_bytes; }

    virtual BBox getBBox( const glm::mat4 &ctm );

private:

    Grid m_grid;
    int m_bytes;
    ParticleGridNode *m_data;
    float m_nodeVolume;

    void colorizeWithMass( const ParticleGridNode &node, float &r, float &g, float &b, float &a ) const;
    void colorizeWithVelocity( const ParticleGridNode &node, float &r, float &g, float &b, float &a ) const;

    GLuint m_vbo;
    int m_vboSize;

    bool hasVBO() const;
    void buildVBO();
    void deleteVBO();

};

#endif // GRIDDATAVIEWER_H
