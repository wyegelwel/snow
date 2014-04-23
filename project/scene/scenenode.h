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

    // A scene node should be deleted through its parent using this
    // function (unless it's the root node) so that the parent
    // doesn't have a dangling NULL pointer
    void deleteChild( SceneNode *child );

    SceneNode* parent() { return m_parent; }

    QList<SceneNode*> getChildren() { return m_children; }

    bool hasRenderable() const { return m_renderable != NULL; }
    void setRenderable( Renderable *renderable );
    Renderable* getRenderable() { return m_renderable; }

    // Render the node's renderable if it is opaque
    virtual void renderOpaque();
    // Render the node's renderable if it is transparent
    virtual void renderTransparent();

    glm::mat4 getCTM();
    // Indicate that the CTM needs recomputing
    void setCTMDirty();

    void applyTransformation( const glm::mat4 &transform );

    // World space bounding box
    BBox getBBox();
    // Indicate that the world space bounding box needs recomputing
    void setBBoxDirty() { m_bboxDirty = true; }

    // World space centroid
    vec3 getCentroid();
    // Indicate that the world space centroid needs recomputing
    void setCentroidDirty() { m_centroidDirty = true; }

    Type getType() { return m_type; }

    // For now, only scene grid nodes are transparent;
    bool isTransparent() const { return m_type == SIMULATION_GRID; }

private:

    SceneNode* m_parent;

    // The following member variables depend on the scene node's
    // cumulative transformation, so they are cached and only
    // recomputed when necessary, if they are labeled "dirty".
    glm::mat4 m_ctm;
    bool m_ctmDirty;
    BBox m_bbox;
    bool m_bboxDirty;
    vec3 m_centroid;
    bool m_centroidDirty;

    glm::mat4 m_transform;

    QList<SceneNode*> m_children;
    Renderable* m_renderable;

    Type m_type;
};

#endif // SCENENODE_H
