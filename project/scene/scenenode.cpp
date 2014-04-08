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
#include "scene/renderable.h"

SceneNode::SceneNode( SceneNode *parent )
    : m_parent(parent),
      m_transform(1.f) // identity matrix
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
