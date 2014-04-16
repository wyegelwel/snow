/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mainwindow.cpp
**   Author: mliberma
**   Created: 6 Apr 2014
**
**************************************************************************/

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "ui/userinput.h"
#include <QFileDialog>
#include <QDir>
#include <iostream>
#include "scene/scene.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
    UserInput::deleteInstance();
}

void MainWindow::loadFromFile()
{
    // pause sim
    ui->viewPanel->pauseDrawing();

    QString fname = QFileDialog::getOpenFileName(this, QString("Open Scene"), QString());
    std::cout << fname.toStdString() << std::endl;
    //ui->viewPanel->loadFromFile(fname);

}

void MainWindow::saveToFile()
{
    QString fname = QFileDialog::getSaveFileName(this, QString("Save Scene"), QString());
    //ui->viewPanel->savetoFile(fname);
}

void MainWindow::renderOffline()
{
    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::Directory);
    QString dirname = dialog.getOpenFileName(this,QString("Output Directory"));

    ui->viewPanel->renderOffline(dirname);
}
