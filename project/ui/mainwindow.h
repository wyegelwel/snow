/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mainwindow.h
**   Author: mliberma
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
    /// restarts simulation and exports to offline renderer.
    void renderOffline();

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    
private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
