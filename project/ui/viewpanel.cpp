/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   viewpanel.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 6 Apr 2014
**
**************************************************************************/

#include <GL/glew.h>
#include <GL/gl.h>

#include <QQueue>

#include "ui/viewpanel.h"

#include "common/common.h"
#include "ui/userinput.h"
#include "viewport/viewport.h"
#include "io/objparser.h"
#include "io/sceneio.h"
#include "geometry/mesh.h"
#include "geometry/bbox.h"
#include "scene/scene.h"
#include "scene/scenegrid.h"
#include "scene/scenenode.h"
#include "scene/scenenodeiterator.h"
#include "sim/engine.h"
#include "sim/particlesystem.h"
#include "ui/infopanel.h"
#include "ui/picker.h"
#include "ui/tools/Tools.h"
#include "ui/uisettings.h"
#include "sim/collider.h"

#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <QFileDialog>
#include <QMessageBox>

#define FPS 30

#define MAJOR_GRID_N 10
#define MAJOR_GRID_TICK 0.5
#define MINOR_GRID_TICK 0.1

ViewPanel::ViewPanel( QWidget *parent )
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
      m_infoPanel(NULL),
      m_tool(NULL)
{
    m_viewport = new Viewport;
    resetViewport();

    m_infoPanel = new InfoPanel(this);
    m_infoPanel->setInfo( "Major Grid Unit", QString::number(MAJOR_GRID_TICK) + " m");
    m_infoPanel->setInfo( "Minor Grid Unit", QString::number(100*MINOR_GRID_TICK) + " cm");
    m_infoPanel->setInfo( "FPS", "XXXXXX" );
    m_infoPanel->setInfo( "Sim Time", "XXXXXXX" );
    m_draw = true;
    m_fps = FPS;

    m_sceneIO = new SceneIO;

    m_scene = new Scene;
    m_engine = new Engine;

    makeCurrent();
    glewInit();
}

ViewPanel::~ViewPanel()
{
    makeCurrent();
    deleteGridVBO();
    SAFE_DELETE( m_engine );
    SAFE_DELETE( m_viewport );
    SAFE_DELETE( m_tool );
    SAFE_DELETE( m_infoPanel );
    SAFE_DELETE( m_scene );
    SAFE_DELETE( m_sceneIO );
}

void
ViewPanel::resetViewport()
{
    m_viewport->orient( glm::vec3( 1, 1, 1 ),
                        glm::vec3( 0, 0, 0 ),
                        glm::vec3( 0, 1, 0 ) );
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

//
    m_infoPanel->setInfo( "Particles", 0 );

    // Render ticker
    assert( connect(&m_ticker, SIGNAL(timeout()), this, SLOT(update())) );
    m_ticker.start( 1000/FPS );
    m_timer.start();
}

void
ViewPanel::teapotDemo()
{
    /// TODO - this function is temporary and written for convenience
    /// in the future, open() and save() will read/write from a file format

    // call load on teapot
    SceneNode *node = new SceneNode;
    QList<Mesh*> meshes;
    OBJParser::load( PROJECT_PATH "/data/models/teapot.obj", meshes );
    // select teapot renderable so we can fill it
    node->setRenderable( meshes[0] );
    m_scene->root()->addChild( node );
    // call fillSelectedMesh()

    ParticleSystem *particles = new ParticleSystem;
    meshes[0]->fill( *particles, 32*512, 0.1f, 200.f );
    m_engine->addParticleSystem( *particles );
    delete particles;

    BBox box = meshes[0]->getBBox( glm::mat4(1.f) );
    UiSettings::gridPosition() = box.min();
    UiSettings::gridDimensions() = glm::ivec3( 128, 128, 128 );
    UiSettings::gridResolution() = box.longestDimSize() / 128.f;
}

void
ViewPanel::paintGL()
{
    glClearColor( 0.20f, 0.225f, 0.25f, 1.f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glPushAttrib( GL_TRANSFORM_BIT );
    glEnable( GL_NORMALIZE );

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

void
ViewPanel::keyPressEvent( QKeyEvent *event )
{
    if ( event->key() == Qt::Key_Backspace ) {
        m_scene->deleteSelectedNodes();
        event->accept();
    } else {
        event->setAccepted( false );
    }
    if ( m_tool ) m_tool->update();
    update();
}

bool ViewPanel::startSimulation()
{
    makeCurrent();
    QString fprefix;
    if ( !m_engine->isRunning() ) {
        m_engine->clearColliders();
        for ( SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it ) {
            if ( (*it)->hasRenderable() ) {
                if ( (*it)->getType() == SceneNode::SIMULATION_GRID ) {
                    m_engine->setGrid( UiSettings::buildGrid((*it)->getCTM()) );
                } else if ( (*it)->getType() == SceneNode::IMPLICIT_COLLIDER ) {
                    Collider collider = *(dynamic_cast<Collider*>((*it)->getRenderable()));
                    glm::mat4 ctm = (*it)->getCTM();
                    m_engine->addCollider(collider,ctm);
                }
            }
        }

        if ( UiSettings::exportVolume() ) {
            fprefix = QFileDialog::getSaveFileName(this, QString("Choose Export Name"), QString());
            if ( fprefix.isEmpty() ) {
                // cancel
                QMessageBox msgBox;
                msgBox.setText("Error : Invalid Volume Export Path");
                msgBox.exec();
                return false;
            }
            m_engine->initExporter(fprefix);
        }

        return m_engine->start( UiSettings::exportVolume() );
    }

    return false;
}

void ViewPanel::stopSimulation()
{
    m_engine->stop();
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
    m_engine->reset();
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

    clearSelection();

    for ( int i = 0; i < meshes.size(); ++i ) {
        Mesh *mesh = meshes[i];
        mesh->setSelected( true );
        mesh->setType( Mesh::SNOW_CONTAINER );
        SceneNode *node = new SceneNode( SceneNode::SNOW_CONTAINER );
        node->setRenderable( mesh );
        m_scene->root()->addChild( node );
    }

    m_tool->update();

    if ( !UiSettings::showContainers() ) emit showMeshes();

}

void ViewPanel::addCollider(ColliderType c,QString planeType)  {
    //TODO add a collider to the scene and set it as selected renderable.
    ImplicitCollider *collider = new ImplicitCollider;
    vec3 parameter;
    SceneNode *node = new SceneNode( SceneNode::IMPLICIT_COLLIDER );
    glm::mat4 transform;glm::vec3 scale;float r;
    switch(c)  {
        case SPHERE:
            r = Collider::SphereRadius();
            parameter = vec3(r,0,0);
            scale = glm::vec3(r,r,r);
            transform = glm::scale( glm::mat4(1.f), scale );
            node->applyTransformation(transform);
            break;
        case HALF_PLANE:
            parameter = vec3(0,1,0);
            break;
        default:
            break;
    }
    Collider *col = new Collider(*collider,c,parameter);


    node->setRenderable( col );
    m_scene->root()->addChild( node );
//    ImplicitCollider &ic = *(col->getImplicitCollider());
//    m_engine->addCollider(ic);

    clearSelection();

    col->setSelected(true);

    m_tool->update();
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
    case Tool::ROTATE:
        m_tool = new RotateTool(this);
        break;
    case Tool::SCALE:
        m_tool = new ScaleTool(this);
        break;
    }
    if ( m_tool ) m_tool->update();
    update();
}

void
ViewPanel::clearSelection()
{
    for ( SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() ) {
            (*it)->getRenderable()->setSelected( false );
        }
    }
}

void ViewPanel::updateSceneGrid()
{
    SceneNode *gridNode = m_scene->getSceneGridNode();
    if ( gridNode ) {
        SceneGrid *grid = dynamic_cast<SceneGrid*>( gridNode->getRenderable() );
        grid->setGrid( UiSettings::buildGrid(glm::mat4(1.f)) );
        gridNode->setBBoxDirty();
        gridNode->setCentroidDirty();
    }
    if ( m_tool ) m_tool->update();
    update();
}

// Paint grid on XZ plane to orient viewport
void
ViewPanel::paintGrid()
{
    if ( !hasGridVBO() ) buildGridVBO();

    glPushAttrib( GL_COLOR_BUFFER_BIT );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_LINE_SMOOTH );
    glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
    glBindBuffer( GL_ARRAY_BUFFER, m_gridVBO );
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, sizeof(vec3), (void*)(0) );
    glColor4f( 0.5f, 0.5f, 0.5f, 0.8f );
    glLineWidth( 2.5f );
    glDrawArrays( GL_LINES, 0, 4 );
    glColor4f( 0.5f, 0.5f, 0.5f, 0.65f );
    glLineWidth( 1.5f );
    glDrawArrays( GL_LINES, 4, m_majorSize-4 );
    glColor4f( 0.5f, 0.5f, 0.5f, 0.5f );
    glLineWidth( 0.5f );
    glDrawArrays( GL_LINES, m_majorSize, m_minorSize );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glDisableClientState( GL_VERTEX_ARRAY );
    glEnd();
    glPopAttrib();
}

bool
ViewPanel::hasGridVBO() const
{
    return m_gridVBO > 0 && glIsBuffer( m_gridVBO );
}

void
ViewPanel::buildGridVBO()
{
    deleteGridVBO();

    QVector<vec3> data;

    static const int minorN = MAJOR_GRID_N * MAJOR_GRID_TICK / MINOR_GRID_TICK;
    static const float max = MAJOR_GRID_N * MAJOR_GRID_TICK;
    for ( int i = 0; i <= MAJOR_GRID_N; ++i ) {
        float x = MAJOR_GRID_TICK * i;
        data += vec3( x, 0.f, -max );
        data += vec3( x, 0.f, max );
        data += vec3( -max, 0.f, x );
        data += vec3( max, 0.f, x );
        if ( i ) {
            data += vec3( -x, 0.f, -max );
            data += vec3( -x, 0.f, max );
            data += vec3( -max, 0.f, -x );
            data += vec3( max, 0.f, -x );
        }
    }
    m_majorSize = data.size();

    for ( int i = -minorN; i <= minorN; ++i ) {
        float x = MINOR_GRID_TICK * i;
        data += vec3( x, 0.f, -max );
        data += vec3( x, 0.f, max );
        data += vec3( -max, 0.f, x );
        data += vec3( max, 0.f, x );
    }
    m_minorSize = data.size() - m_majorSize;

    glGenBuffers( 1, &m_gridVBO );
    glBindBuffer( GL_ARRAY_BUFFER, m_gridVBO );
    glBufferData( GL_ARRAY_BUFFER, data.size()*sizeof(vec3), data.data(), GL_STATIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
}

void
ViewPanel::deleteGridVBO()
{
    if ( hasGridVBO() ) {
        glDeleteBuffers( 1, &m_gridVBO );
    }
    m_gridVBO = 0;
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

        makeCurrent();

        ParticleSystem *particles = new ParticleSystem;
        mesh->fill( *particles, UiSettings::fillNumParticles(), UiSettings::fillResolution(), UiSettings::fillDensity() );
        m_engine->addParticleSystem( *particles );
        delete particles;

        m_infoPanel->setInfo( "Particles", QString::number(m_engine->particleSystem()->size()) );
    }

    delete mesh;

    if ( !UiSettings::showParticles() ) emit showParticles();
}

void
ViewPanel::saveSelectedMesh()
{
    QList<Mesh*> meshes;

    for ( SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() &&
             (*it)->getType() == SceneNode::SNOW_CONTAINER &&
             (*it)->getRenderable()->isSelected() ) {
            Mesh *copy = new Mesh( *dynamic_cast<Mesh*>((*it)->getRenderable()) );
            copy->applyTransformation( (*it)->getCTM() );
            meshes += copy;
        }
    }

    // If there's a mesh selection, save it
    if ( !meshes.empty() )  {
        QString filename = QFileDialog::getSaveFileName( this, "Choose mesh file destination.", PROJECT_PATH "/data/models/" );
        if ( !filename.isNull() ) {
            if ( OBJParser::save( filename, meshes ) ) {
                for ( int i = 0; i < meshes.size(); ++i )
                    delete meshes[i];
                meshes.clear();
                LOG( "Mesh saved to %s", STR(filename) );
            }
        }
    }


}

void
ViewPanel::applyMaterials()
{
    // re-apply particleSystem
    m_engine->initParticleMaterials(UiSettings::materialPreset());
}

void
ViewPanel::loadScene()
{
    // call file dialog
    QString str;
    m_sceneIO->read(str, m_scene, m_engine);
}

void
ViewPanel::saveScene()
{
    // this is also called when exporting.
    QString str;
    m_sceneIO->write(str, m_scene, m_engine);
}

