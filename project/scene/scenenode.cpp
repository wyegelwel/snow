/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scenenode.cpp
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#include <GL/gl.h>
#include <glm/gtc/type_ptr.hpp>

#include "scene/scenenode.h"
#include "common/renderable.h"

SceneNode::SceneNode( SceneNode *parent )
    : m_parent(parent),
      m_transform(1.f) // identity matrix
{
}

SceneNode::SceneNode(SceneNodeType type, QString objfile)
    : m_type(type),
      m_objfile(objfile)
{
}

SceneNode::~SceneNode()
{
    clearChildren();
    clearRenderables();
}

void
SceneNode::clearChildren()
{
    for ( int i = 0; i < m_children.size(); ++i )
        delete m_children[i];
    m_children.clear();
}

void
SceneNode::addChild( SceneNode *child )
{
    m_children += child;
    child->m_parent = this;
    child->m_ctmDirty = true;
}

void
SceneNode::clearRenderables()
{
    for ( int i = 0; i < m_renderables.size(); ++i )
        delete m_renderables[i];
    m_renderables.clear();
}

void
SceneNode::addRenderable( Renderable *renderable )
{
    m_renderables += renderable;
}

void
SceneNode::render()
{
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glMultMatrixf( glm::value_ptr(m_transform) );
    for ( int i = 0; i < m_renderables.size(); ++i )
        m_renderables[i]->render();
    for ( int i = 0; i < m_children.size(); ++i )
        m_children[i]->render();
    glPopMatrix();
}

glm::mat4
SceneNode::getCTM()
{
    if ( m_ctmDirty ) {
        glm::mat4 pCtm = ( m_parent ) ? m_parent->getCTM() : glm::mat4();
        m_ctm = pCtm * m_transform;
        m_ctmDirty = false;
    }
    return m_ctm;
}

QString
SceneNode::getObjFile()
{
    return m_objfile;
}

SceneNodeType
SceneNode::getType()
{
    return m_type;
}
