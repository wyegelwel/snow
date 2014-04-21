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
#include <QPixmap>
#include <iostream>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "ui/userinput.h"
#include "scene/scene.h"

#include "ui/databinding.h"
#include "ui/uisettings.h"
#include "ui/viewpanel.h"
#include "ui/tools/tool.h"

#include "sim/collider.h"

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


void MainWindow::importMesh()
{
    ui->viewPanel->pauseSimulation();
    ui->viewPanel->pauseDrawing();

    QString filename = QFileDialog::getOpenFileName(this, "Select mesh to import.", PROJECT_PATH "/data/models", "*.obj");
    if ( !filename.isEmpty() ) {
        ui->viewPanel->loadMesh( filename );
    }
    ui->showMeshCheckbox->setChecked( true );

    ui->viewPanel->resumeSimulation();
    ui->viewPanel->resumeDrawing();
}

void MainWindow::addCollider()  {
    ui->viewPanel->pauseSimulation();
    ui->viewPanel->pauseDrawing();

    QString colliderType = ui->chooseCollider->currentText();
    bool isType = true;
    ColliderType c;
    if(colliderType == "Sphere") {
        c = SPHERE;
    }
    else if(colliderType == "Vertical Plane")  {
        c = HALF_PLANE;
    }
    else if(colliderType == "Horizontal Plane")  {
        c = HALF_PLANE;
    }
    else {isType = false;}
    if(isType)  {
        ui->viewPanel->addCollider(c,colliderType);
    }
    ui->viewPanel->resumeSimulation();
    ui->viewPanel->resumeDrawing();
}

void MainWindow::setupUI()
{
    // Mesh Filling
    assert( connect(ui->importButton, SIGNAL(clicked()), this, SLOT(importMesh())) );
    assert( connect(ui->fillButton, SIGNAL(clicked()), ui->viewPanel, SLOT(fillSelectedMesh())) );
    FloatBinding::bindSpinBox( ui->fillResolutionSpinbox, UiSettings::fillResolution(), this );
    IntBinding::bindSpinBox( ui->fillNumParticlesSpinbox, UiSettings::fillNumParticles(), this );

    // Simulation
    assert( connect(ui->startButton, SIGNAL(clicked()), ui->viewPanel, SLOT(startSimulation())) );
    assert( connect(ui->pauseButton, SIGNAL(toggled(bool)), ui->viewPanel, SLOT(pauseSimulation(bool))) );
    assert( connect(ui->gridXSpinbox, SIGNAL(valueChanged(int)), ui->viewPanel, SLOT(updateSceneGrid())) );
    assert( connect(ui->gridYSpinbox, SIGNAL(valueChanged(int)), ui->viewPanel, SLOT(updateSceneGrid())) );
    assert( connect(ui->gridZSpinbox, SIGNAL(valueChanged(int)), ui->viewPanel, SLOT(updateSceneGrid())) );
    assert( connect(ui->gridResolutionSpinbox, SIGNAL(valueChanged(double)), ui->viewPanel, SLOT(updateSceneGrid())) );
    IntBinding::bindSpinBox( ui->gridXSpinbox, UiSettings::gridDimensions().x, this );
    IntBinding::bindSpinBox( ui->gridYSpinbox, UiSettings::gridDimensions().y, this );
    IntBinding::bindSpinBox( ui->gridZSpinbox, UiSettings::gridDimensions().z, this );
    FloatBinding::bindSpinBox( ui->gridResolutionSpinbox, UiSettings::gridResolution(), this );

    BoolBinding::bindCheckBox( ui->exportCheckbox, UiSettings::exportSimulation(), this );

    // Collider

    // Connect buttons to slots
    assert( connect(ui->colliderAddButton, SIGNAL(clicked()), this, SLOT(addCollider())));

    // Connect values to settings - not sure how to do this with combo box.

    // Scene

    // Connect buttons to slots
    assert( connect(ui->saveSceneButton, SIGNAL(clicked()), this, SLOT(saveToFile())));
    assert( connect(ui->loadSceneButton, SIGNAL(clicked()), this, SLOT(loadFromFile())));
    assert( connect(ui->editSimConstantsButton, SIGNAL(clicked()), ui->viewPanel, SLOT(editSnowConstants())));

    // View Panel
    assert( connect(ui->showGridCheckbox, SIGNAL(toggled(bool)), ui->showGridCombo, SLOT(setEnabled(bool))) );
    assert( connect(ui->showMeshCheckbox, SIGNAL(toggled(bool)), ui->showMeshCombo, SLOT(setEnabled(bool))) );
    CheckboxBoolAttribute::bindBool( ui->showMeshCheckbox, &UiSettings::showMesh(), this );
    ComboIntAttribute::bindInt( ui->showMeshCombo, &UiSettings::showMeshMode(), this );
    CheckboxBoolAttribute::bindBool( ui->showGridCheckbox, &UiSettings::showGrid(), this );
    ComboIntAttribute::bindInt( ui->showGridCombo, &UiSettings::showGridMode(), this );
    CheckboxBoolAttribute::bindBool( ui->showGridDataCheckbox, &UiSettings::showGridData(), this );
    ComboIntAttribute::bindInt( ui->showGridDataCombo, &UiSettings::showGridDataMode(), this );
    CheckboxBoolAttribute::bindBool( ui->showParticlesCheckbox, &UiSettings::showParticles(), this );
    ComboIntAttribute::bindInt( ui->showParticlesCombo, &UiSettings::showParticlesMode(), this );

    // Tools
    ui->toolButtonGroup->setId( ui->selectionToolButton, Tool::SELECTION );
    ui->toolButtonGroup->setId( ui->moveToolButton, Tool::MOVE );
    assert( connect(ui->toolButtonGroup, SIGNAL(buttonClicked(int)), ui->viewPanel, SLOT(setTool(int))) );
    ui->selectionToolButton->click();
}

void MainWindow::resizeEvent( QResizeEvent* )
{
    UiSettings::windowSize() = size();
}

void MainWindow::moveEvent( QMoveEvent* )
{
    UiSettings::windowPosition() = pos();
}

void MainWindow::takeScreenshot()
{
    ui->viewPanel->pauseDrawing();
    ui->viewPanel->pauseSimulation();

    QPixmap pixmap(this->rect().size());
    this->render(&pixmap, QPoint(), QRegion(this->rect()));
    // prompt user where to save it

    QString fname = QFileDialog::getSaveFileName(this, QString("Save Screenshot"), PROJECT_PATH "/data/");
    if ( !fname.isEmpty() ) {
        QFile file(fname);
        file.open(QIODevice::WriteOnly);
        pixmap.save(&file, "PNG");
    }
    ui->viewPanel->resumeDrawing();
    ui->viewPanel->resumeSimulation();
}
