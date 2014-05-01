/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scene.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 8 Apr 2014
**
**************************************************************************/

#include <GL/gl.h>

#include <QQueue>

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/mat4x4.hpp"
#include "glm/vec4.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "scene.h"

#include "common/common.h"
#include "scene/scenegrid.h"
#include "scene/scenenode.h"
#include "scene/scenenodeiterator.h"
#include "ui/uisettings.h"

Scene::Scene()
    : m_root(new SceneNode)
{
    // Add scene grid
    SceneNode *gridNode = new SceneNode( SceneNode::SIMULATION_GRID );
    Grid grid;
    grid.pos = vec3(0,0,0);
    grid.dim = UiSettings::gridDimensions();
    grid.h = UiSettings::gridResolution();
    gridNode->setRenderable( new SceneGrid(grid) );
    glm::mat4 transform = glm::translate( glm::mat4(1.f), glm::vec3(UiSettings::gridPosition().x,UiSettings::gridPosition().y,UiSettings::gridPosition().z) );
    gridNode->applyTransformation( transform );
    m_root->addChild( gridNode );
}

Scene::~Scene()
{
    for ( SceneNodeIterator it = begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() && (*it)->getType() == SceneNode::SIMULATION_GRID ) {
            glm::vec4 point = (*it)->getCTM() * glm::vec4(0,0,0,1);
            UiSettings::gridPosition() = vec3(point.x, point.y, point.z);
            break;
        }
    }
    UiSettings::saveSettings();
    SAFE_DELETE( m_root );
}

void
Scene::render()
{
    setupLights();
    // Render opaque objects, then overlay with transparent objects
    m_root->renderOpaque();
    m_root->renderTransparent();
}

void
Scene::renderVelocity(bool velTool) {
    m_root->renderVelocity(velTool);
}

void
Scene::setupLights()
{
    glm::vec4 diffuse = glm::vec4( 0.5f, 0.5f, 0.5f, 1.f );
    for ( int i = 0; i < 5; ++i ) {
        glEnable( GL_LIGHT0 + i );
        glLightfv( GL_LIGHT0 + i, GL_DIFFUSE, glm::value_ptr(diffuse) );
    }

    glLightfv( GL_LIGHT0, GL_POSITION, glm::value_ptr(glm::vec4(100.f, 0.f, 0.f, 1.f)) );
    glLightfv( GL_LIGHT1, GL_POSITION, glm::value_ptr(glm::vec4(-100.f, 0.f, 0.f, 1.f)) );
    glLightfv( GL_LIGHT2, GL_POSITION, glm::value_ptr(glm::vec4(0.f, 0.f, 100.f, 1.f)) );
    glLightfv( GL_LIGHT3, GL_POSITION, glm::value_ptr(glm::vec4(0.f, 0.f, -100.f, 1.f)) );
    glLightfv( GL_LIGHT4, GL_POSITION, glm::value_ptr(glm::vec4(0.f, 100.f, 0.f, 1.f)) );
}

SceneNodeIterator
Scene::begin() const
{
    QList<SceneNode*> nodes;
    nodes += m_root;
    int i = 0;
    while ( i < nodes.size() ) {
        nodes += nodes[i]->getChildren();
        i++;
    }
    return SceneNodeIterator( nodes );
}

SceneNode*
Scene::getSceneGridNode()
{
    for ( int i = 0; i < m_root->getChildren().size(); ++i ) {
        SceneNode *child = m_root->getChildren()[i];
        if ( child->hasRenderable() && (child->getType() == SceneNode::SIMULATION_GRID) ) {
            return child;
        }
    }
    return NULL;
}

// Note: don't use iterator here, because we want to make sure we don't
// try and delete nodes twice (i.e., if it was already deleted because
// its parent was deleted). Also we don't allow for the SceneGrid node to
// be deleted.
void
Scene::deleteSelectedNodes()
{
    QQueue<SceneNode*> nodes;
    nodes += m_root;
    while ( !nodes.empty() ) {
        SceneNode *node = nodes.dequeue();
        if ( node->hasRenderable() && node->getType() != SceneNode::SIMULATION_GRID && node->getRenderable()->isSelected() ) {
            // Delete node through its parent so that the scene graph is appropriately
            // rid of the deleted node.
            node->parent()->deleteChild( node );
        } else {
            nodes += node->getChildren();
        }
    }
}
