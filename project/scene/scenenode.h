/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scenenode.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef SCENENODE_H
#define SCENENODE_H

#include <QList>
#include <QString>

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/mat4x4.hpp"

#include "geometry/bbox.h"

class Renderable;

class SceneNode
{

public:

    enum Type
    {
        TRANSFORM,
        IMPLICIT_COLLIDER,
        SNOW_CONTAINER,
        SIMULATION_GRID
    };

    SceneNode( Type type = TRANSFORM );
    virtual ~SceneNode();

    void clearChildren();
    void addChild( SceneNode *child );
    void deleteChild( SceneNode *child );

    SceneNode* parent() { return m_parent; }

    QList<SceneNode*> getChildren() { return m_children; }

    bool hasRenderable() const { return m_renderable != NULL; }
    void setRenderable( Renderable *renderable );
    Renderable* getRenderable() { return m_renderable; }

    virtual void renderOpaque();
    virtual void renderTransparent();

    glm::mat4 getCTM();
    void setCTMDirty();

    void applyTransformation( const glm::mat4 &transform );

    BBox getBBox();
    void setBBoxDirty() { m_bboxDirty = true; }

    Type getType() { return m_type; }

    // For now, only scene grid nodes are transparent;
    bool isTransparent() const { return m_type == SIMULATION_GRID; }

private:

    SceneNode* m_parent;

    glm::mat4 m_ctm;
    bool m_ctmDirty;

    BBox m_bbox;
    bool m_bboxDirty;

    glm::mat4 m_transform;

    QList<SceneNode*> m_children;
    Renderable* m_renderable;

    Type m_type;
};

#endif // SCENENODE_H
