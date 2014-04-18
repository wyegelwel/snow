/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mainwindow.cpp
**   Author: mliberma
**   Created: 6 Apr 2014
**
**************************************************************************/

#include <QFileDialog>
#include <QDir>
#include <iostream>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "ui/userinput.h"
#include "scene/scene.h"

#include "ui/databinding.h"
#include "ui/uisettings.h"
#include "ui/viewpanel.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    UiSettings::loadSettings();
    ui->setupUi(this);
    setupUI();
}

MainWindow::~MainWindow()
{
    UiSettings::saveSettings();
    UserInput::deleteInstance();
    delete ui;
}

void MainWindow::loadFromFile()
{
    // pause sim
    ui->viewPanel->pause();
    QDir sceneDir("../project/data/scenes");
    sceneDir.makeAbsolute();
    QString fname = QFileDialog::getOpenFileName(this, QString("Open Scene"), sceneDir.absolutePath());
    std::cout << fname.toStdString() << std::endl;
    //ui->viewPanel->loadFromFile(fname);
}

void MainWindow::saveToFile()
{
    ui->viewPanel->pause();
    QDir sceneDir("../project/data/scenes");
    sceneDir.makeAbsolute();
    QString fname = QFileDialog::getSaveFileName(this, QString("Save Scene"), sceneDir.absolutePath());
    //ui->viewPanel->saveToFile(fname);
    ui->viewPanel->resume();
}

void MainWindow::renderOffline()
{
    ui->viewPanel->pause();
    QDir sceneDir("~/offline_renders");
    sceneDir.makeAbsolute();
    QString fprefix = QFileDialog::getSaveFileName(this, QString("Save Scene"), sceneDir.absolutePath());
    ui->viewPanel->renderOffline(fprefix);
}

void MainWindow::importMesh()
{
    ui->viewPanel->pause();
    QString filename = QFileDialog::getOpenFileName(this, "Select mesh to import.", PROJECT_PATH "/data/models", "*.obj");
    if ( !filename.isEmpty() ) {
        ui->viewPanel->generateNewMesh(filename);
    }
    ui->viewPanel->resume();
}

void MainWindow::setupUI()
{
    // Mesh Filling

    // Connect buttons to slots
    assert( connect(ui->importButton, SIGNAL(clicked()), this, SLOT(importMesh())) );
    assert( connect(ui->fillButton, SIGNAL(clicked()), ui->viewPanel, SLOT(fillSelectedMesh())) );

    // Connect values to settings
    FloatBinding::bindSpinBox( ui->fillResolutionSpinbox, UiSettings::fillResolution(), this );
    IntBinding::bindSpinBox( ui->fillNumParticlesSpinbox, UiSettings::fillNumParticles(), this );
}
