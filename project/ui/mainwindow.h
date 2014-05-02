/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mainwindow.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 6 Apr 2014
**
**************************************************************************/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public slots:

    void takeScreenshot();
    void fillNumParticleFinishedEditing();

    void importMesh();
    void addCollider();

//    void setVelMagText(float f,bool b);
//    void setVelVecText(vec3,bool);
    void setVelText(bool b,float f,float x,float y,float z);
    void setSelectionText(QString,bool,int);

    void startSimulation();
    void stopSimulation();


    virtual void resizeEvent( QResizeEvent* );
    virtual void moveEvent( QMoveEvent* );

    virtual void keyPressEvent( QKeyEvent *event );


public:

    explicit MainWindow( QWidget *parent = 0 );
    ~MainWindow();

private:

    Ui::MainWindow *ui;

    void setupUI();

};

#endif // MAINWINDOW_H
