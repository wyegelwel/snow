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
#include "sim/collider.h"

class InfoPanel;
class Viewport;
class Scene;
class Engine;
class SceneNode;
class Tool;
class SelectionTool;

class ViewPanel : public QGLWidget
{
    Q_OBJECT

public:

    ViewPanel( QWidget *parent );
    virtual ~ViewPanel();

    // not implemented
    void saveToFile(QString fname);
    void loadFromFile(QString fname);

public slots:

    void resetViewport();

    virtual void initializeGL();
    virtual void paintGL();

    virtual void resizeEvent( QResizeEvent *event );

    virtual void mousePressEvent( QMouseEvent *event );
    virtual void mouseMoveEvent( QMouseEvent *event );
    virtual void mouseReleaseEvent( QMouseEvent *event );
    virtual void keyPressEvent( QKeyEvent *event );

    void startSimulation();
    void pauseSimulation( bool pause = true );
    void resumeSimulation();
    void resetSimulation();

    void pauseDrawing();
    void resumeDrawing();

    // Filling
    void fillSelectedMesh();
    void loadMesh( const QString &filename );

    void addCollider(ColliderType c);

    void editSnowConstants();

    void setTool( int tool );

    void updateSceneGrid();

    // Demo Scenes
    void teapotDemo();

protected:

    QTimer m_ticker;
    QElapsedTimer m_timer;

    InfoPanel *m_infoPanel;
    Viewport *m_viewport;
    Tool *m_tool;

    Engine *m_engine;
    Scene *m_scene;

    bool m_draw;

    float m_fps;

    void paintGrid();

    friend class SelectionTool;
    friend class MoveTool;

};

#endif // VIEWPANEL_H
