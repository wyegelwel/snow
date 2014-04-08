/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scene.cpp
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#include "scene.h"

#include "common/common.h"
#include "scene/scenenode.h"

Scene::Scene()
    : m_root(new SceneNode(NULL))
{
}

Scene::~Scene()
{
    SAFE_DELETE( m_root );
}

void
Scene::render()
{
    if ( m_root ) m_root->render();
}
