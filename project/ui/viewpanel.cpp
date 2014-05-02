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
#include "scene/scenecollider.h"
#include "scene/scenegrid.h"
#include "scene/scenenode.h"
#include "scene/scenenodeiterator.h"
#include "sim/engine.h"
#include "sim/implicitcollider.h"
#include "sim/particlesystem.h"
#include "ui/infopanel.h"
#include "ui/picker.h"
#include "ui/tools/Tools.h"
#include "ui/uisettings.h"
#include "ui/tools/velocitytool.h"

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include <glm/gtx/string_cast.hpp>

#include <QFileDialog>
#include <QMessageBox>

#define FPS 30

#define MAJOR_GRID_N 2
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
ViewPanel::paintGL()
{
    glClearColor( 0.20f, 0.225f, 0.25f, 1.f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glPushAttrib( GL_TRANSFORM_BIT );
    glEnable( GL_NORMALIZE );

//    bool velTool;
//    if((m_tool->getType()) == VelocityTool) {
//        velTool = true;
//    }

    m_viewport->push(); {
        m_scene->render();
        m_scene->renderVelocity(true);
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


    float currTime = m_engine->getSimulationTime();
    updateColliders(currTime - m_prevTime);
    m_prevTime = currTime;


    m_infoPanel->render();

    glPopAttrib();
}

void ViewPanel::updateColliders(float timestep) {
    for ( SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() ) {
            if ( (*it)->getType() == SceneNode::SCENE_COLLIDER ) {
                SceneCollider* c = dynamic_cast<SceneCollider*>((*it)->getRenderable());
                glm::vec3 v = c->getWorldVelVec((*it)->getCTM());
//                glm::vec4 newV = (*it)->getCTM()*glm::vec4(v,1);
                if(!EQ(c->getVelMag(),0))
                    v = glm::normalize(v);
                glm::mat4 transform = glm::translate(glm::mat4(),v*c->getVelMag()*timestep);
                (*it)->applyTransformation(transform);
            }
        }
    }
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
                } else if ( (*it)->getType() == SceneNode::SCENE_COLLIDER ) {
                    SceneCollider *sceneCollider = dynamic_cast<SceneCollider*>((*it)->getRenderable());
                    ImplicitCollider &collider( *(sceneCollider->getImplicitCollider()) );
                    glm::mat4 ctm = (*it)->getCTM();
                    collider.applyTransformation( ctm );

                    glm::vec3 v = (*it)->getRenderable()->getWorldVelVec(ctm);
                    collider.velocity = (*it)->getRenderable()->getVelMag()*v;
                    m_engine->addCollider( collider );
                }
            }
        }

        const bool exportVol = UiSettings::exportDensity() || UiSettings::exportVelocity();
        if ( exportVol ) {
            bool ok = !m_sceneIO->sceneFile().isNull();
            if (!ok) // have not saved yet
                ok = saveScene();
            if (ok)
                m_engine->initExporter(m_sceneIO->sceneFile());
        }

        return m_engine->start(exportVol);
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

    checkSelected();
    if ( !UiSettings::showContainers() ) emit showMeshes();

}

void ViewPanel::addCollider(int colliderType)  {
    vec3 parameter;
    SceneNode *node = new SceneNode( SceneNode::SCENE_COLLIDER );
    float r;
    switch ( (ColliderType)colliderType ) {
        case SPHERE:
            r = SceneCollider::SphereRadius();
            parameter = vec3( r, 0, 0 );
            node->applyTransformation( glm::scale( glm::mat4(1.f), glm::vec3(r,r,r) ) );
            break;
        case HALF_PLANE:
            parameter = vec3(0,1,0);
            break;
        default:
            break;
    }

    ImplicitCollider *collider = new ImplicitCollider( (ColliderType)colliderType, vec3(0,0,0), parameter, vec3(0,0,0), 0.2f );
    SceneCollider *sceneCollider = new SceneCollider( collider );

    node->setRenderable( sceneCollider );
    glm::mat4 ctm = node->getCTM();
    sceneCollider->setCTM(ctm);
    m_scene->root()->addChild( node );

    clearSelection();
    sceneCollider->setSelected( true );

    m_tool->update();
    checkSelected();
}

void ViewPanel::checkSelected()  {
   int counter = 0;
   for ( SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it ) {
       if ( (*it)->hasRenderable() &&(*it)->getRenderable()->isSelected()) {
           counter++;
           m_selected = (*it);
       }
   }
   if(counter == 0)  {
       emit changeVel(false);
       emit changeSelection("Currently Selected: none",false);
       m_selected = NULL;
   }
   else if(counter == 1 && m_selected->getType() != SceneNode::SIMULATION_GRID)  {
       glm::vec3 v;// = glm::vec3(m_selected->getRenderable()->getVelVec());
//       glm::vec4 vWorld = m_selected->getCTM()*glm::vec4((v),1);
//       float mag = glm::length(v);
       if EQ(m_selected->getRenderable()->getVelMag(), 0)
       {
           emit changeVel(true,m_selected->getRenderable()->getVelMag(),0,0,0);
       }
       else
       {
//           v = glm::normalize(glm::vec3(vWorld.x,vWorld.y,vWorld.z));
           v=m_selected->getRenderable()->getWorldVelVec(m_selected->getCTM());
           emit changeVel(true,m_selected->getRenderable()->getVelMag(),v.x,v.y,v.z);
       }
       emit changeSelection("Currently Selected: ",true,m_selected->getType());
   }
   else if(counter == 1 && m_selected->getType() == SceneNode::SIMULATION_GRID)  {
       emit changeVel(false);
       emit changeSelection("Currently Selected: Grid",false);
   }
   else  {
       emit changeVel(false);
       emit changeSelection("Currently Selected: more than one object",false);
       m_selected = NULL;
   }
}

void ViewPanel::setTool( int tool )
{
    SAFE_DELETE( m_tool );
    Tool::Type t = (Tool::Type)tool;
    switch ( t ) {
    case Tool::SELECTION:
        m_tool = new SelectionTool(this,t);
        break;
    case Tool::MOVE:
        m_tool = new MoveTool(this,t);
        break;
    case Tool::ROTATE:
        m_tool = new RotateTool(this,t);
        break;
    case Tool::SCALE:
        m_tool = new ScaleTool(this,t);
        break;
    case Tool::VELOCITY:
        m_tool = new VelocityTool(this,t);
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
    checkSelected();
}

void ViewPanel::updateSceneGrid()
{
//    SceneNode *gridNode = m_scene->getSceneGridNode();
//    if ( gridNode ) {
//        SceneGrid *grid = dynamic_cast<SceneGrid*>( gridNode->getRenderable() );
//        grid->setGrid( UiSettings::buildGrid(glm::mat4(1.f)) );
//        gridNode->setBBoxDirty();
//        gridNode->setCentroidDirty();
//    }
    m_scene->updateSceneGrid();
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
    glm::vec3 currentVel;
    float currentMag;
    for ( SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() &&
             (*it)->getType() == SceneNode::SNOW_CONTAINER &&
             (*it)->getRenderable()->isSelected() ) {
            Mesh * original = dynamic_cast<Mesh*>((*it)->getRenderable());

            original->setParticleCount(UiSettings::fillNumParticles());
            original->setMaterialPreset(UiSettings::materialPreset());

            Mesh *copy = new Mesh( *original );
            const glm::mat4 transformation = (*it)->getCTM();
            copy->applyTransformation( transformation );
            mesh->append( *copy );
            delete copy;

            currentVel = (*it)->getRenderable()->getWorldVelVec(transformation);
            currentMag = (*it)->getRenderable()->getVelMag();
            if(EQ(0,currentMag)) {
                currentVel = vec3(0,0,0);
            }
            else  {
                currentVel = vec3(currentVel.x,currentVel.y,currentVel.z);
            }
        }
    }

    // If there's a selection, do mesh->fill...
    if ( !mesh->isEmpty() )  {

        makeCurrent();

        ParticleSystem *particles = new ParticleSystem;
        particles->setVelMag(currentMag);
        particles->setVelVec(currentVel);
        mesh->fill( *particles, UiSettings::fillNumParticles(), UiSettings::fillResolution(), UiSettings::fillDensity() );
        particles->setVelocity();
        m_engine->addParticleSystem( *particles );
        delete particles;

        m_infoPanel->setInfo( "Particles", QString::number(m_engine->particleSystem()->size()) );
    }

    delete mesh;

    if ( !UiSettings::showParticles() ) emit showParticles();
}

void ViewPanel::giveVelToSelected() {
    if(!m_selected) return;
    for ( SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() &&
             (*it)->getType() != SceneNode::SIMULATION_GRID &&
             (*it)->getRenderable()->isSelected() ) {
            (*it)->getRenderable()->setVelMag(1.0f);
            (*it)->getRenderable()->setVelVec(glm::vec3(0,1,0));
            (*it)->getRenderable()->updateMeshVel();
        }
    }
    checkSelected();
}

void ViewPanel::zeroVelOfSelected()  {
    if(!m_selected) return;
    for ( SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() &&
             (*it)->getType() != SceneNode::SIMULATION_GRID &&
             (*it)->getRenderable()->isSelected() ) {
            (*it)->getRenderable()->setVelMag(0.0f);
            (*it)->getRenderable()->setVelVec(glm::vec3(0,0,0));
            (*it)->getRenderable()->updateMeshVel();
        }
    }
    checkSelected();
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

bool
ViewPanel::openScene()
{
    pauseDrawing();
    // call file dialog
    QString filename = QFileDialog::getOpenFileName(this, "Choose Scene File Path", PROJECT_PATH "/data/scenes/");
    if (!filename.isNull())
        m_sceneIO->read(filename, m_scene, m_engine);
    else
        LOG("could not open file \n");
    resumeDrawing();
}

bool
ViewPanel::saveScene()
{
    pauseDrawing();
    // this is enforced if engine->start is called and export is not checked
    QString foo = m_sceneIO->sceneFile();
    if (m_sceneIO->sceneFile().isNull())
    {
        // filename not initialized yet
        QString filename = QFileDialog::getSaveFileName( this, "Choose Scene File Path", PROJECT_PATH "/data/scenes/" );

        if (!filename.isNull())
        {
            m_sceneIO->setSceneFile(filename);
            m_sceneIO->write(m_scene, m_engine);
        }
        else
        {
            QMessageBox msgBox;
            msgBox.setText("Error : Invalid Save Path");
            msgBox.exec();
            return false;
        }
    }
    else
    {
        m_sceneIO->write(m_scene, m_engine);
    }
    resumeDrawing();
}

