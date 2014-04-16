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

class InfoPanel;
class Viewport;
class Scene;
class ParticleSystem;

class ViewPanel : public QGLWidget
{

    Q_OBJECT

public:

    ViewPanel( QWidget *parent );
    virtual ~ViewPanel();

    void pauseDrawing();
    void resumeDrawing();
    void saveToFile(QString fname);
    void loadFromFile(QString fname);
    void renderOffline(QString outputDir);

public slots:

    void resetViewport();

    virtual void initializeGL();
    virtual void paintGL();

    virtual void resizeEvent( QResizeEvent *event );

    virtual void mousePressEvent( QMouseEvent *event );
    virtual void mouseMoveEvent( QMouseEvent *event );
    virtual void mouseReleaseEvent( QMouseEvent *event );

private:

    QTimer m_ticker;
    QElapsedTimer m_timer;

    InfoPanel *m_infoPanel;
    Viewport *m_viewport;

    ParticleSystem *m_particles;
    Scene *m_scene;

    // set true to draw a little XYZ axis in the corner
    bool m_drawAxis;
};

#endif // VIEWPANEL_H
