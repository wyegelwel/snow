/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scenenode.h
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef SCENENODE_H
#define SCENENODE_H

#include <QList>
#include <glm/mat4x4.hpp>
#include <QString>

class Renderable;

// we could make the type a part of renderable (i.e. make sub-part of OBJ a collider)
// but that would make the offline rendering pipeline kind of messy. leaving it like this for now.
enum SceneNodeType{
    IMPLICIT_COLLIDER, SNOW_CONTAINER, OBJ
};

class SceneNode
{

public:

    SceneNode( SceneNode *parent = NULL );
    SceneNode(SceneNodeType type, QString objfile);
    virtual ~SceneNode();

    void clearChildren();
    void addChild( SceneNode *child );

    void clearRenderables();
    void addRenderable( Renderable *renderable );

    /// traverses children, collapses all scene nodes into a flat list.
    QList<SceneNode*> allNodes();

    virtual void render();

    glm::mat4 getCTM();

    SceneNodeType getType();
    QString getObjFile();

private:

    SceneNode* m_parent;

    glm::mat4 m_ctm;
    bool m_ctmDirty;

    glm::mat4 m_transform;

    QList<SceneNode*> m_children;
    QList<Renderable*> m_renderables;

    SceneNodeType m_type;
    QString m_objfile;
};

#endif // SCENENODE_H
