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

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/vec4.hpp"
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
    grid.pos = UiSettings::gridPosition();
    grid.dim = UiSettings::gridDimensions();
    grid.h = UiSettings::gridResolution();
    gridNode->setRenderable( new SceneGrid(grid) );
    m_root->addChild( gridNode );
}

Scene::~Scene()
{
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
Scene::setupLights()
{
    glm::vec4 diffuse = glm::vec4( 0.125f, 0.125f, 0.125f, 1.f );
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
