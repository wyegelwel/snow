/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   viewpanel.cpp
**   Author: mliberma
**   Created: 6 Apr 2014
**
**************************************************************************/

#include <GL/gl.h>

#include "viewpanel.h"

#include <glm/gtc/random.hpp>

#include "common/common.h"
#include "ui/userinput.h"
#include "viewport/viewport.h"
#include "io/objparser.h"
#include "geometry/mesh.h"
#include "scene/scene.h"
#include "scene/scenenode.h"
#include "sim/engine.h"
#include "sim/particle.h"
#include "ui/infopanel.h"
#include "ui/uisettings.h"

/// TEMPORARY
#include "io/sceneparser.h"
#include "io/mitsubaexporter.h"

#define FPS 30

ViewPanel::ViewPanel( QWidget *parent )
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
      m_infoPanel(NULL)
{
    m_viewport = new Viewport;
    resetViewport();

    m_infoPanel = new InfoPanel(this);
    m_infoPanel->setInfo( "FPS", "XXXXXX" );
    m_drawAxis = true;

    m_scene = new Scene;
    m_engine = new Engine;
    m_scene->setParticleSystem( m_engine->particleSystem() );
}

ViewPanel::~ViewPanel()
{
    SAFE_DELETE( m_engine );
    SAFE_DELETE( m_viewport );
    SAFE_DELETE( m_infoPanel );
    SAFE_DELETE( m_scene );
}

void
ViewPanel::resetViewport()
{
    m_viewport->orient( glm::vec3( 0, 5, 12.5),
                        glm::vec3(  0,  0,  0),
                        glm::vec3(  0,  1,  0) );
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
    // OpenGL states

    QGLWidget::initializeGL();

    glEnable( GL_DEPTH_TEST );
    glDepthFunc( GL_LESS );

    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    glEnable( GL_LINE_SMOOTH );
    glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );

    // Scene

    SceneNode *node = new SceneNode;

    QList<Mesh*> meshes;
    OBJParser::load( PROJECT_PATH "/data/models/teapot.obj", meshes );
    for ( int i = 0; i < meshes.size(); ++i )
        node->addRenderable( meshes[i] );
    m_scene->root()->addChild( node );

    this->makeCurrent();

    ParticleSystem *particles = new ParticleSystem;
    meshes[0]->fill( *particles, 128*512, 0.1f );
    m_engine->addParticleSystem( *particles );
    delete particles;

    m_engine->grid().dim = glm::ivec3( 128, 128, 128 );

    BBox box = meshes[0]->getObjectBBox();
    box.expandRel( .2f );
    m_engine->grid().pos = box.min();
    m_engine->grid().h = box.longestDimSize() / 256.f;

    m_infoPanel->setInfo( "Particles", QString::number(m_engine->particleSystem()->size()) );

    // Render ticker
    assert( connect(&m_ticker, SIGNAL(timeout()), this, SLOT(update())) );
    m_ticker.start( 1000/FPS );
    m_timer.start();

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

    float fps = 1000.f / m_timer.restart();
    m_infoPanel->setInfo( "FPS", QString::number(fps, 'f', 2), false );
    m_infoPanel->render();
}

void
ViewPanel::mousePressEvent( QMouseEvent *event )
{
    UserInput::update(event);
    if ( UserInput::ctrlKey() ) {
        if ( UserInput::leftMouse() ) m_viewport->setState( Viewport::TUMBLING );
        else if ( UserInput::rightMouse() ) m_viewport->setState( Viewport::ZOOMING );
        else if ( UserInput::middleMouse() ) m_viewport->setState( Viewport::PANNING );
    }
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


// UI methods to pause/resume drawing and simulation
// we might want to do this when doing non-viewing stuff, like opening new files
void ViewPanel::pause()
{
    m_engine->pause();
}

void ViewPanel::resume()
{
    m_engine->resume();
}


/// temporary hack: I'm calling the SceneParser from here for the file saving
/// and offline rendering. Ideally this would be handled by the Engine class.
void ViewPanel::saveToFile(QString fname)
{
    // write - not done, do this out later
    SceneParser::write(fname, m_scene);
}

void ViewPanel::loadFromFile(QString fname)
{
    // read - not done, figure this out later
    SceneParser::read(fname, m_scene);
}

void ViewPanel::renderOffline(QString file_prefix)
{
    /**
     * the exporter handles scene by scene so here, we tell the simulation to start over
     * then call exportScene every frame
     */
    reset();
    // step the simulation 1/24 of a second at a time.

//    for (int s=0; s<1; s++)
//    {
//        for (int f=0; f<24; f++)
//        {
//            MitsubaExporter::exportScene(file_prefix, f, m_scene);
//        }
//    }

    // for now, just export the first frame
    MitsubaExporter::exportScene(file_prefix, 0, m_scene);
}

void ViewPanel::start()
{
    m_engine->start();
}

void ViewPanel::reset()
{

}

void ViewPanel::fillSelectedMesh()
{
    // If there's a selection, do mesh->fill...
    int nParticles = UiSettings::fillNumParticles();
    float resolution = UiSettings::fillResolution();
    // mesh->fill( *(m_engine->particleSystem()), nParticles, resolution );
}
