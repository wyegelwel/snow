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
    QDir sceneDir("../project/data/scenes");
    sceneDir.makeAbsolute();
    QString fname = QFileDialog::getOpenFileName(this, QString("Open Scene"), sceneDir.absolutePath());
    std::cout << fname.toStdString() << std::endl;
    //ui->viewPanel->loadFromFile(fname);
}

void MainWindow::saveToFile()
{
    ui->viewPanel->pauseDrawing();
    QDir sceneDir("../project/data/scenes");
    sceneDir.makeAbsolute();
    QString fname = QFileDialog::getSaveFileName(this, QString("Save Scene"), sceneDir.absolutePath());
    //ui->viewPanel->saveToFile(fname);
    ui->viewPanel->resumeDrawing();
}

void MainWindow::renderOffline()
{
    ui->viewPanel->pauseDrawing();
    QDir sceneDir("~/offline_renders");
    sceneDir.makeAbsolute();
    QString fprefix = QFileDialog::getSaveFileName(this, QString("Save Scene"), sceneDir.absolutePath());
    ui->viewPanel->renderOffline(fprefix);
}

