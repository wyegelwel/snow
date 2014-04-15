/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   engine.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#include "engine.h"
#include "sim/AppSettings.h"
#include "scene/scene.h"
#include "sim/particle.h"

Engine::Engine()
{
    m_scene = new Scene;
    m_paused = false;

//    SceneNode *node = new SceneNode;

//    QList<Mesh*> meshes;
//    OBJParser::load( PROJECT_PATH "/data/models/teapot.obj", meshes );
//    for ( int i = 0; i < meshes.size(); ++i )
//        node->addRenderable( meshes[i] );

//    m_particles = new ParticleSystem;
//    meshes[0]->fill( *m_particles, 256*512, 0.1f );
//    node->addRenderable( m_particles );

//    m_scene->root()->addChild( node );
//    m_numParticles = m_particles->particles().size();

}


void Engine::load(QFile file)
{
    // load the scene XML file
}


void Engine::start()
{
    // run the simulation
}

void Engine::exportMitsuba()
{

}

int Engine::numParticles()
{
    return m_particles->particles().size();
}

//float t = 0.f;
void Engine::update()
{
    // engine keeps track of its own time.
    // Somehow
}
