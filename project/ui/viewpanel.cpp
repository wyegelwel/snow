/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   viewpanel.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 6 Apr 2014
**
**************************************************************************/

#include <GL/gl.h>
#include <QQueue>

#include "viewpanel.h"

#include <glm/gtc/random.hpp>

#include "common/common.h"
#include "ui/userinput.h"
#include "viewport/viewport.h"
#include "io/objparser.h"
#include "geometry/mesh.h"
#include "geometry/bbox.h"
#include "scene/scene.h"
#include "scene/scenenode.h"
#include "scene/scenenodeiterator.h"
#include "sim/engine.h"
#include "sim/particle.h"
#include "ui/infopanel.h"
#include "ui/picker.h"
#include "ui/uisettings.h"
#include "sim/collider.h"

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
    m_infoPanel->setInfo( "Sim Time", "XXXXXXX" );
    m_draw = true;
    m_fps = FPS;

    m_scene = new Scene;
    m_engine = new Engine;
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
    m_viewport->orient( glm::vec3( 0, 5, 12.5 ),
                        glm::vec3( 0, 1,    0 ),
                        glm::vec3( 0, 1,    0 ) );
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
    meshes[0]->fill( *particles, 32*512, 0.1f );
    m_engine->addParticleSystem( *particles );
    delete particles;

    Grid grid;
    grid.dim = glm::ivec3( 128, 128, 128 );
    BBox box = meshes[0]->getWorldBBox( glm::mat4(1.f) );
    grid.pos = box.min();
    grid.h = box.longestDimSize() / 128.f;

    m_engine->setGrid( grid );

    m_infoPanel->setInfo( "Particles", QLocale().toString(m_engine->particleSystem()->size()) );

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
        m_engine->render();
        m_viewport->drawAxis();
    } m_viewport->pop();

    if ( m_draw ) {
        static const float filter = 0.8f;
        m_fps = (1-filter)*(1000.f/MAX(m_timer.restart(),1)) + filter*m_fps;
        m_infoPanel->setInfo( "FPS", QString::number(m_fps, 'f', 2), false );
        m_infoPanel->setInfo( "Sim Time", QString::number(m_engine->getSimulationTime(), 'f', 3)+" s", false );
    }

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
    } else {
        if ( UserInput::leftMouse() ) {
            clearSelection();
            m_viewport->loadPickMatrices( UserInput::mousePos() );
            Renderable *clicked = getClickedRenderable();
            if ( clicked ) {
                clicked->setSelected( true );
            }
            m_viewport->popMatrices();
        }
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

/// temporary hack: I'm calling the SceneParser from here for the file saving
/// and offline rendering. Ideally this would be handled by the Engine class.
/// do this after the MitsubaExporter is working
void ViewPanel::saveToFile(QString fname)
{
    // write - not done, figure this out later
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

    resetSimulation();

    // step the simulation 1/24 of a second at a time.

//    for (int s=0; s<1; s++)
//    {
//        for (int f=0; f<24; f++)
//        {
//            MitsubaExporter::exportScene(file_prefix, f, m_scene, cam);
//        }
//    }

    // for now, just export the first frame

    MitsubaExporter exporter;
    exporter.exportScene(file_prefix, 0);
}

void ViewPanel::startSimulation()
{
    m_engine->start();
}

void ViewPanel::pauseSimulation( bool pause )
{
    if ( pause ) m_engine->pause();
    else m_engine->resume();
}

void ViewPanel::resetSimulation()
{
    LOG( "NOT YET IMPLEMENTED." );
}

void ViewPanel::pauseDrawing()
{
    m_ticker.stop();
    m_draw = false;
}

void ViewPanel::resumeDrawing()
{
    m_ticker.start( 1000/FPS );
    m_draw = true;
}

void ViewPanel::generateNewMesh( const QString &f )
{
// single obj file is associated with multiple renderables and a single
// scene node.
    SceneNode *node = new SceneNode(SNOW_CONTAINER, f);

    QList<Mesh*> meshes;
    OBJParser::load(f, meshes );
    for ( int i = 0; i < meshes.size(); ++i )
        node->addRenderable( meshes[i] );

    m_scene->root()->addChild( node );
    m_selectedMesh = meshes[0];
}

void ViewPanel::fillSelectedMesh()
{
    // If there's a selection, do mesh->fill...
    if ( m_selectedMesh )  {
        int nParticles = UiSettings::fillNumParticles();
        float resolution = UiSettings::fillResolution();
        ParticleSystem *particles = new ParticleSystem;
        m_selectedMesh->fill( *particles, nParticles, resolution );
        m_engine->addParticleSystem( *particles );
        delete particles;
        m_infoPanel->setInfo( "Particles", QString::number(m_engine->particleSystem()->size()) );
    }
}

Renderable* ViewPanel::getClickedRenderable()
{
    QList<Renderable*> renderables;
    for ( SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it ) {
        renderables += (*it)->getRenderables();
    }
    if ( !renderables.empty() ) {
        Picker picker( renderables.size() );
        for ( int i = 0; i < renderables.size(); ++i ) {
            picker.setObjectIndex(i);
            renderables[i]->renderForPicker();
        }
        unsigned int index = picker.getPick();
        if ( index != Picker::NO_PICK ) {
            return renderables[index];
        }
    }
    return NULL;
}

void ViewPanel::clearSelection()
{
    QList<Renderable*> renderables;
    for ( SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it ) {
        renderables += (*it)->getRenderables();
    }
    for ( int i = 0; i < renderables.size(); ++i ) {
        renderables[i]->setSelected( false );
    }
}
