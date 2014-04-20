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

#include "common/common.h"
#include "ui/userinput.h"
#include "viewport/viewport.h"
#include "io/objparser.h"
#include "geometry/mesh.h"
#include "geometry/bbox.h"
#include "scene/scene.h"
#include "scene/scenegrid.h"
#include "scene/scenenode.h"
#include "scene/scenenodeiterator.h"
#include "sim/engine.h"
#include "sim/particle.h"
#include "ui/infopanel.h"
#include "ui/picker.h"
#include "ui/tools/Tools.h"
#include "ui/uisettings.h"
#include "sim/collider.h"

#include <QFileDialog>


#define FPS 30

ViewPanel::ViewPanel( QWidget *parent )
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
      m_infoPanel(NULL),
      m_tool(NULL)
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
    SAFE_DELETE( m_tool );
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

//    SceneNode *node = new SceneNode;

//    QList<Mesh*> meshes;
//    OBJParser::load( PROJECT_PATH "/data/models/teapot.obj", meshes );
//    node->setRenderable( meshes[0] );
//    m_scene->root()->addChild( node );

//    ParticleSystem *particles = new ParticleSystem;
//    meshes[0]->fill( *particles, 32*512, 0.1f );
//    m_engine->addParticleSystem( *particles );
//    delete particles;

//    Grid grid;
//    grid.dim = glm::ivec3( 128, 128, 128 );
//    BBox box = meshes[0]->getWorldBBox( glm::mat4(1.f) );
//    grid.pos = box.min();
//    grid.h = box.longestDimSize() / 128.f;

//    m_engine->setGrid( grid );

    m_infoPanel->setInfo( "Particles", 0 );

    // Render ticker
    assert( connect(&m_ticker, SIGNAL(timeout()), this, SLOT(update())) );
    m_ticker.start( 1000/FPS );
    m_timer.start();

}

float t = 0.f;

void
ViewPanel::paintGL()
{
    glClearColor( 0.20f, 0.225f, 0.25f, 1.f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    m_viewport->push(); {
        m_scene->render();
        m_engine->render();
        paintGrid();
        if ( m_tool ) m_tool->render();
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

// Paint grid on XZ plane to orient viewport
void
ViewPanel::paintGrid()
{
    glLineWidth( 1.f );
    glPushAttrib( GL_COLOR_BUFFER_BIT );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glColor4f( 0.5f, 0.5f, 0.5f, 0.5f );
    glBegin( GL_LINES );
    for ( int i = -10; i <= 10; ++i ) {
        glVertex3f( i, 0.f, -10.f );
        glVertex3f( i, 0.f, 10.f );
        glVertex3f( -10.f, 0.f, i );
        glVertex3f( 10.f, 0.f, i );
    }
    glEnd();
    glPopAttrib();
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
        if ( UserInput::leftMouse() ) if ( m_tool ) m_tool->mousePressed();
    }
    update();
}

void
ViewPanel::mouseMoveEvent( QMouseEvent *event )
{
    UserInput::update(event);
    m_viewport->mouseMoved();
    if ( m_tool ) m_tool->mouseMoved();
    update();
}

void
ViewPanel::mouseReleaseEvent( QMouseEvent *event )
{
    UserInput::update(event);
    m_viewport->setState( Viewport::IDLE );
    if ( m_tool ) m_tool->mouseReleased();
    update();
}

/// temporary hack: I'm calling the SceneParser from here for the file saving
/// and offline rendering. Ideally this would be handled by the Engine class.
/// do this after the MitsubaExporter is working
void ViewPanel::saveToFile(QString fname)
{
    // write - not done, figure this out later
    //SceneParser::write(fname, m_scene);
}

void ViewPanel::loadFromFile(QString fname)
{
    // read - not done, figure this out later
    //SceneParser::read(fname, m_scene);
}


void ViewPanel::startSimulation()
{
    if ( !m_engine->isRunning() && UiSettings::exportSimulation() )
    {
        // ask the user where the data should be saved
//        QDir sceneDir("~/offline_renders");
//        sceneDir.makeAbsolute();
        QString fprefix = QFileDialog::getSaveFileName(this, QString("Choose Export Name"), QString());
        m_engine->initExporter(fprefix);
    }
    m_engine->setGrid( UiSettings::buildGrid() );
    m_engine->start( UiSettings::exportSimulation() );
}

void ViewPanel::pauseSimulation( bool pause )
{
    if ( pause ) m_engine->pause();
    else m_engine->resume();
}

void ViewPanel::resumeSimulation()
{
    m_engine->resume();
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

void ViewPanel::loadMesh( const QString &filename )
{

    // single obj file is associated with multiple renderables and a single
    // scene node.
    QList<Mesh*> meshes;
    OBJParser::load( filename, meshes );

    for ( int i = 0; i < meshes.size(); ++i ) {
        SceneNode *node = new SceneNode( SceneNode::SNOW_CONTAINER );
        node->setRenderable( meshes[i] );
        m_scene->root()->addChild( node );
    }

}

void ViewPanel::fillSelectedMesh()
{
    Mesh *mesh = new Mesh;

    for ( SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() &&
             (*it)->getType() == SceneNode::SNOW_CONTAINER &&
             (*it)->getRenderable()->isSelected() ) {
            Mesh *copy = new Mesh( *dynamic_cast<Mesh*>((*it)->getRenderable()) );
            const glm::mat4 transformation = (*it)->getCTM();
            copy->applyTransformation( transformation );
            mesh->append( *copy );
            delete copy;
        }
    }

    // If there's a selection, do mesh->fill...
    if ( !mesh->isEmpty() )  {

        int nParticles = UiSettings::fillNumParticles();
        float resolution = UiSettings::fillResolution();

        ParticleSystem *particles = new ParticleSystem;
        mesh->fill( *particles, nParticles, resolution );
        m_engine->addParticleSystem( *particles );
        delete particles;

        m_infoPanel->setInfo( "Particles", QString::number(m_engine->particleSystem()->size()) );
    }

    delete mesh;
}

void ViewPanel::addCollider(ColliderType c)  {
    //TODO add a collider to the scene and set it as selected renderable.
}

void ViewPanel::editSnowConstants()  {
    //TODO create popout to edit snow constants? Other (possibly better) option is to have all
    // constants listed in UI in LineEdits and have them editable, then we could bind it to UISettings.
}

void ViewPanel::setTool( int tool )
{
    SAFE_DELETE( m_tool );
    switch ( (Tool::Type)tool ) {
    case Tool::SELECTION:
        m_tool = new SelectionTool(this);
        break;
    case Tool::MOVE:
        m_tool = new MoveTool(this);
        break;
    default:
        break;
    }
    if ( m_tool ) m_tool->update();
    update();
}

void ViewPanel::updateSceneGrid()
{
    SceneNode *gridNode = m_scene->getSceneGridNode();
    if ( gridNode ) {
        SceneGrid *grid = dynamic_cast<SceneGrid*>( gridNode->getRenderable() );
        grid->setGrid( UiSettings::buildGrid() );
    }
    update();
}
