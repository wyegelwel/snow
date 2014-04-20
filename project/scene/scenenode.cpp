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

#include "common/common.h"
#include "common/renderable.h"
#include "scene/scenenode.h"

SceneNode::SceneNode( Type type )
    : m_parent(NULL),
      m_ctm(1.f),
      m_ctmDirty(true),
      m_transform(1.f),
      m_renderable(NULL),
      m_type(type)
{
}

SceneNode::~SceneNode()
{
    SAFE_DELETE( m_renderable );
    clearChildren();
}

void
SceneNode::clearChildren()
{
    for ( int i = 0; i < m_children.size(); ++i )
        SAFE_DELETE( m_children[i] );
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
SceneNode::setRenderable( Renderable *renderable )
{
    SAFE_DELETE( m_renderable );
    m_renderable = renderable;
}

void
SceneNode::render()
{
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glMultMatrixf( glm::value_ptr(m_transform) );
    if ( m_renderable ) m_renderable->render();
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
