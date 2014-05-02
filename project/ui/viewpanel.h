/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   viewpanel.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 6 Apr 2014
**
**************************************************************************/

#ifndef VIEWPANEL_H
#define VIEWPANEL_H

#include <QGLWidget>
#include <QTimer>
#include <QElapsedTimer>
#include <QFile>
#include <QDir>
#include "geometry/mesh.h"

class InfoPanel;
class Viewport;
class Scene;
class Engine;
class SceneNode;
class Tool;
class SelectionTool;
class SceneIO;

class ViewPanel : public QGLWidget
{
    Q_OBJECT

public:

    ViewPanel( QWidget *parent );
    virtual ~ViewPanel();

    // not implemented
    void saveToFile(QString fname);
    void loadFromFile(QString fname);

    // Returns whether or not it started
    bool startSimulation();
    void stopSimulation();

public slots:

    void resetViewport();

    virtual void initializeGL();
    virtual void paintGL();

    virtual void resizeEvent( QResizeEvent *event );

    virtual void mousePressEvent( QMouseEvent *event );
    virtual void mouseMoveEvent( QMouseEvent *event );
    virtual void mouseReleaseEvent( QMouseEvent *event );
    virtual void keyPressEvent( QKeyEvent *event );

    void resetSimulation();
    void pauseSimulation( bool pause = true );
    void resumeSimulation();
    void pauseDrawing();
    void resumeDrawing();
    void updateColliders(float timestep);

    void loadMesh( const QString &filename );

    void addCollider(int colliderType);

    void setTool( int tool );

    void updateSceneGrid();

    void clearSelection();
    void fillSelectedMesh();
    void saveSelectedMesh();

    void openScene();
    void saveScene();

    void zeroVelOfSelected();
    void giveVelToSelected();
    void checkSelected();

signals:

    void showMeshes();
    void showParticles();
//    void changeVelMag(float f,bool b);
    void changeSelection(QString s,bool b,int i=0);
//    void changeVelVec(vec3 &v,bool b);
    void changeVel(bool b,float f=0,float x=0,float y=0,float z=0);

protected:

    QTimer m_ticker;
    QElapsedTimer m_timer;

    InfoPanel *m_infoPanel;
    Viewport *m_viewport;
    Tool *m_tool;

    SceneIO * m_sceneIO;

    Engine *m_engine;
    Scene *m_scene;

    GLuint m_gridVBO;
    int m_majorSize;
    int m_minorSize;
    bool m_draw;
    float m_fps;
    float m_prevTime;

    void paintGrid();

    bool hasGridVBO() const;
    void buildGridVBO();
    void deleteGridVBO();
    void addParticleSystem(ParticleSystem &particles);

    friend class Tool;
    friend class SelectionTool;
    friend class MoveTool;
    friend class RotateTool;
    friend class ScaleTool;
    friend class VelocityTool;

    SceneNode *m_selected;

    //friend class SceneIO;

};

#endif // VIEWPANEL_H
