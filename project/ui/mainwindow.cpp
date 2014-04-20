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

    this->setWindowTitle( "SNOW" );
    this->move( UiSettings::windowPosition() );
    this->resize( UiSettings::windowSize() );
}

MainWindow::~MainWindow()
{
    UiSettings::saveSettings();
    UserInput::deleteInstance();
    delete ui;
}

void MainWindow::loadFromFile()
{
    ui->viewPanel->pauseSimulation();
    ui->viewPanel->pauseDrawing();

    QDir sceneDir("../project/data/scenes");
    sceneDir.makeAbsolute();
    QString fname = QFileDialog::getOpenFileName(this, QString("Open Scene"), sceneDir.absolutePath());
    std::cout << fname.toStdString() << std::endl;
    //ui->viewPanel->loadFromFile(fname);

    ui->viewPanel->resumeSimulation();
    ui->viewPanel->resumeDrawing();
}

void MainWindow::saveToFile()
{
    ui->viewPanel->pauseSimulation();
    ui->viewPanel->pauseDrawing();

    QDir sceneDir("../project/data/scenes");
    sceneDir.makeAbsolute();
    QString fname = QFileDialog::getSaveFileName(this, QString("Save Scene"), sceneDir.absolutePath());
    //ui->viewPanel->saveToFile(fname);

    ui->viewPanel->resumeSimulation();
    ui->viewPanel->resumeDrawing();
}

void MainWindow::renderOffline()
{
    ui->viewPanel->pauseSimulation();
    ui->viewPanel->pauseDrawing();

    QDir sceneDir("~/offline_renders");
    sceneDir.makeAbsolute();
    QString fprefix = QFileDialog::getSaveFileName(this, QString("Choose Export Filename"), sceneDir.absolutePath());
    ui->viewPanel->renderOffline(fprefix);

    ui->viewPanel->resumeSimulation();
    ui->viewPanel->resumeDrawing();
}

void MainWindow::importMesh()
{
    ui->viewPanel->pauseSimulation();
    ui->viewPanel->pauseDrawing();

    QString filename = QFileDialog::getOpenFileName(this, "Select mesh to import.", PROJECT_PATH "/data/models", "*.obj");
    if ( !filename.isEmpty() ) {
        ui->viewPanel->generateNewMesh(filename);
    }

    ui->viewPanel->resumeSimulation();
    ui->viewPanel->resumeDrawing();
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

    // Simulation

    // Connect button to slots
    assert( connect(ui->startButton, SIGNAL(clicked()), ui->viewPanel, SLOT(startSimulation())) );
    assert( connect(ui->pauseButton, SIGNAL(toggled(bool)), ui->viewPanel, SLOT(pauseSimulation(bool))) );

    // Connect values to settings
    BoolBinding::bindCheckBox( ui->exportCheckbox, UiSettings::exportSimulation(), this );

}

void MainWindow::resizeEvent( QResizeEvent* )
{
    UiSettings::windowSize() = size();
}

void MainWindow::moveEvent( QMoveEvent* )
{
    UiSettings::windowPosition() = pos();
}
