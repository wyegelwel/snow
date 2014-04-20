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

    /// saves snow simulation parameters / colliders etc. to file
    void saveToFile();

    /// loads snow simulation parameters from file
    void loadFromFile();

    void importMesh();

    void checkMeshRenderSettings();

    void addCollider();

    virtual void resizeEvent( QResizeEvent* );
    virtual void moveEvent( QMoveEvent* );

public:

    explicit MainWindow( QWidget *parent = 0 );
    ~MainWindow();

private:

    Ui::MainWindow *ui;

    void setupUI();

};

#endif // MAINWINDOW_H
