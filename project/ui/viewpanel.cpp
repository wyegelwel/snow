/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   viewpanel.cpp
**   Author: mliberma
**   Created: 6 Apr 2014
**
**************************************************************************/

#include "viewpanel.h"

#include <glm/gtc/random.hpp>

#include "common/common.h"
#include "ui/userinput.h"
#include "viewport/viewport.h"
#include "io/objparser.h"
#include "scene/mesh.h"
#include "scene/scene.h"
#include "scene/scenenode.h"
#include "sim/particle.h"

#define FPS 60

ViewPanel::ViewPanel( QWidget *parent )
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    m_viewport = new Viewport;
    resetViewport();

    m_drawAxis = true;

    m_scene = new Scene;
    SceneNode *node = new SceneNode;

    QList<Mesh*> meshes;
    OBJParser::load( PROJECT_PATH "/data/models/teapot.obj", meshes );
    for ( int i = 0; i < meshes.size(); ++i )
        node->addRenderable( meshes[i] );

    m_particles = new ParticleSystem;
    for ( int i = 0; i < 4*512; ++i ) {
        Particle particle;
        particle.position = glm::ballRand( 2.5f );
        *m_particles += particle;
    }
    node->addRenderable( m_particles );

    m_scene->root()->addChild( node );

}

ViewPanel::~ViewPanel()
{
    SAFE_DELETE( m_viewport );
    SAFE_DELETE( m_scene );
}

void
ViewPanel::resetViewport()
{
    m_viewport->orient( glm::vec3( 10,10, 10),
                        glm::vec3( 0,  0,  0),
                        glm::vec3( 0,  1,  0) );
    m_viewport->setDimensions( width(), height() );
}

void
ViewPanel::resizeEvent( QResizeEvent *event )
{
    QGLWidget::resizeEvent( event );
    m_viewport->setDimensions( width(), height() );
}

void
ViewPanel::initializeGL()
{
    QGLWidget::initializeGL();

    glEnable( GL_DEPTH_TEST );
    glDepthFunc( GL_LESS );

    assert( connect(&m_timer, SIGNAL(timeout()), this, SLOT(update())) );
    m_timer.start( 1000/FPS );
}

float t = 0.f;

void
ViewPanel::paintGL()
{
    glClearColor( 0.f, 0.f, 0.f, 0.f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    m_viewport->push(); {

        m_scene->render();

        if ( m_drawAxis ) m_viewport->drawAxis();

    } m_viewport->pop();

    m_particles->update( t += 1.f/FPS );
}

void
ViewPanel::mousePressEvent( QMouseEvent *event )
{
    UserInput::update(event);
    if ( UserInput::leftMouse() ) m_viewport->setState( Viewport::TUMBLING );
    else if ( UserInput::rightMouse() ) m_viewport->setState( Viewport::ZOOMING );
    else if ( UserInput::middleMouse() ) m_viewport->setState( Viewport::PANNING );
}

void
ViewPanel::mouseMoveEvent( QMouseEvent *event )
{
    UserInput::update(event);
    m_viewport->mouseMoved();
}

void
ViewPanel::mouseReleaseEvent( QMouseEvent *event )
{
    UserInput::update(event);
    m_viewport->setState( Viewport::IDLE );
}
