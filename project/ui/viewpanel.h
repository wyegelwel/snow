/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   viewpanel.h
**   Author: mliberma
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

class ViewPanel : public QGLWidget
{
    Q_OBJECT

public:

    ViewPanel( QWidget *parent );
    virtual ~ViewPanel();

    void saveToFile(QString fname);
    void loadFromFile(QString fname);
    void renderOffline(QString file_prefix);

public slots:

    void resetViewport();

    virtual void initializeGL();
    virtual void paintGL();

    virtual void resizeEvent( QResizeEvent *event );

    virtual void mousePressEvent( QMouseEvent *event );
    virtual void mouseMoveEvent( QMouseEvent *event );
    virtual void mouseReleaseEvent( QMouseEvent *event );

    void startSimulation();
    void pauseSimulation( bool pause = true );
    void resumeSimulation() { pauseSimulation(false); }
    void resetSimulation();

    void pauseDrawing();
    void resumeDrawing();

    // Filling
    void fillSelectedMesh();

    void generateNewMesh(const QString &f);

    void addCollider(ColliderType c);

    void editSnowConstants();

private:

    QTimer m_ticker;
    QElapsedTimer m_timer;

    InfoPanel *m_infoPanel;
    Viewport *m_viewport;

    Engine *m_engine;
    Scene *m_scene;

    Mesh *m_selectedMesh;

    bool m_draw;

    float m_fps;

};

#endif // VIEWPANEL_H
