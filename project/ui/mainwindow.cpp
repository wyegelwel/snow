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

#include "ui/collapsiblebox.h"
#include "ui/databinding.h"
#include "ui/uisettings.h"
#include "ui/viewpanel.h"
#include "ui/tools/tool.h"

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
    UserInput::deleteInstance();
    delete ui;
    UiSettings::saveSettings();
}

void MainWindow::importMesh()
{
    ui->viewPanel->pauseSimulation();
    ui->viewPanel->pauseDrawing();

    QString filename = QFileDialog::getOpenFileName(this, "Select mesh to import.", PROJECT_PATH "/data/models", "*.obj");
    if ( !filename.isEmpty() ) {
        ui->viewPanel->loadMesh( filename );
    }

    ui->viewPanel->resumeSimulation();
    ui->viewPanel->resumeDrawing();
}

void MainWindow::addCollider()
{
    ui->viewPanel->addCollider( ui->chooseCollider->currentIndex() );
}

void MainWindow::startSimulation()
{
    if ( ui->viewPanel->startSimulation() ) {
        ui->viewPanel->clearSelection();
        ui->selectionToolButton->click();
        ui->startButton->setEnabled( false );
        ui->stopButton->setEnabled( true );
        ui->pauseButton->setEnabled( true );
        ui->resetButton->setEnabled( false );
    }
}

void MainWindow::stopSimulation()
{
    ui->viewPanel->stopSimulation();
    ui->startButton->setEnabled( true );
    ui->stopButton->setEnabled( false );
    if ( ui->pauseButton->isChecked() ) {
        ui->pauseButton->click();
    }
    ui->pauseButton->setEnabled( false );
    ui->resetButton->setEnabled( true );
}

void MainWindow::setVelText(bool b,float f,float x, float y, float z) {
    if(!b)  {
        ui->velLabel->setText("Velocity: ");
        return;
    }
    glm::vec3 v = vec3(x,y,z);
    v = v*f;
    QString toSet = "Velocity: (";

    QString s1 = QString::number(v.x,'g',2);
    toSet.append(s1);
    toSet.append(" m/s,");
    QString s2 = QString::number(v.y,'g',2);
    toSet.append(s2);
    toSet.append(" m/s,");
    QString s3 = QString::number(v.z,'g',2);
    toSet.append(s3);
    toSet.append(" m/s)");
    ui->velLabel->setText(toSet);

}

void MainWindow::setSelectionText(QString s, bool b, int i)  {
    if(!b) {
        ui->currentlySelectedLabel->setText(s);
        return;
    }
    switch(i)  {
    case(1):
        ui->currentlySelectedLabel->setText("Currently Selected: Collider");
        break;
    case(2):
        ui->currentlySelectedLabel->setText("Currently Selected: Snow Container");
        break;
    default:
        break;
    }
}

void MainWindow::takeScreenshot()
{
    // this has issues rasterizing particles...
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
        file.close();
    }
    ui->viewPanel->resumeDrawing();
    ui->viewPanel->resumeSimulation();
}

void MainWindow::fillNumParticleFinishedEditing()
{
    // rounds number of particles to nearest multiple of 512
    const int n = ui->fillNumParticlesSpinbox->value();
    int numParticles = (n/512) * 512;
    numParticles += 512*(n < 512);
    if (numParticles != n)
        ui->fillNumParticlesSpinbox->setValue(numParticles);
}

void MainWindow::setupUI()
{
    assert( connect(ui->actionSave_Mesh, SIGNAL(triggered()), ui->viewPanel, SLOT(saveSelectedMesh())) );
    assert( connect(ui->actionOpen_Scene, SIGNAL(triggered()), ui->viewPanel, SLOT(openScene())));
    assert( connect(ui->actionSave_Scene, SIGNAL(triggered()), ui->viewPanel, SLOT(saveScene())));

    // Mesh Filling
    assert( connect(ui->importButton, SIGNAL(clicked()), this, SLOT(importMesh())) );
    assert( connect(ui->fillButton, SIGNAL(clicked()), ui->viewPanel, SLOT(fillSelectedMesh())) );
    FloatBinding::bindSpinBox( ui->fillResolutionSpinbox, UiSettings::fillResolution(), this );
    IntBinding::bindSpinBox( ui->fillNumParticlesSpinbox, UiSettings::fillNumParticles(), this );
    assert( connect(ui->fillNumParticlesSpinbox, SIGNAL(editingFinished()), this, SLOT(fillNumParticleFinishedEditing())) );
    FloatBinding::bindSpinBox( ui->densitySpinbox, UiSettings::fillDensity(), this );
    ComboIntAttribute::bindInt(ui->snowMaterialCombo, &UiSettings::materialPreset(), this);
    assert( connect(ui->meshGiveVelocityButton, SIGNAL(clicked()), ui->viewPanel,SLOT(giveVelToSelected())));
    assert( connect(ui->MeshZeroVelocityButton, SIGNAL(clicked()), ui->viewPanel,SLOT(zeroVelOfSelected())));

    // Simulation
    assert( connect(ui->startButton, SIGNAL(clicked()), this, SLOT(startSimulation())) );
    assert( connect(ui->stopButton, SIGNAL(clicked()), this, SLOT(stopSimulation())) );
    assert( connect(ui->pauseButton, SIGNAL(toggled(bool)), ui->viewPanel, SLOT(pauseSimulation(bool))) );
    assert( connect(ui->resetButton, SIGNAL(clicked()), ui->viewPanel, SLOT(resetSimulation())) );
    IntBinding::bindSpinBox( ui->gridXSpinbox, UiSettings::gridDimensions().x, this );
    IntBinding::bindSpinBox( ui->gridYSpinbox, UiSettings::gridDimensions().y, this );
    IntBinding::bindSpinBox( ui->gridZSpinbox, UiSettings::gridDimensions().z, this );
    FloatBinding::bindSpinBox( ui->gridResolutionSpinbox, UiSettings::gridResolution(), this );
    assert( connect(ui->gridXSpinbox, SIGNAL(valueChanged(int)), ui->viewPanel, SLOT(updateSceneGrid())) );
    assert( connect(ui->gridYSpinbox, SIGNAL(valueChanged(int)), ui->viewPanel, SLOT(updateSceneGrid())) );
    assert( connect(ui->gridZSpinbox, SIGNAL(valueChanged(int)), ui->viewPanel, SLOT(updateSceneGrid())) );
    assert( connect(ui->gridResolutionSpinbox, SIGNAL(valueChanged(double)), ui->viewPanel, SLOT(updateSceneGrid())) );
    FloatBinding::bindSpinBox( ui->timeStepSpinbox, UiSettings::timeStep(), this );
    BoolBinding::bindCheckBox( ui->implicitCheckbox, UiSettings::implicit(), this );
    //assert( connect(ui->snowMaterialCombo, SIGNAL(currentIndexChanged(int)), ui->viewPanel, SLOT(applyMaterials())) );

    // exporting
    BoolBinding::bindCheckBox( ui->exportDensityCheckbox, UiSettings::exportDensity(), this );
    BoolBinding::bindCheckBox(ui->exportVelocityCheckbox, UiSettings::exportVelocity(), this);
    IntBinding::bindSpinBox(ui->exportFPSSpinBox, UiSettings::exportFPS(), this);
    FloatBinding::bindSpinBox(ui->maxTimeSpinBox, UiSettings::maxTime(),this);

    // SceneCollider
    assert( connect(ui->colliderAddButton, SIGNAL(clicked()), this, SLOT(addCollider())) );
//    assert( connect(ui->ColliderVelocityButton, SIGNAL(clicked()), ui->viewPanel,SLOT(giveVelToSelected())));
//    assert( connect(ui->colliderZeroButton, SIGNAL(clicked()), ui->viewPanel,SLOT(zeroVelOfSelected())));

    // View Panel
    assert( connect(ui->showContainersCheckbox, SIGNAL(toggled(bool)), ui->showContainersCombo, SLOT(setEnabled(bool))) );
    assert( connect(ui->showCollidersCheckbox, SIGNAL(toggled(bool)), ui->showCollidersCombo, SLOT(setEnabled(bool))) );
    assert( connect(ui->showGridCheckbox, SIGNAL(toggled(bool)), ui->showGridCombo, SLOT(setEnabled(bool))) );
    assert( connect(ui->showGridDataCheckbox, SIGNAL(toggled(bool)), ui->showGridDataCombo, SLOT(setEnabled(bool))) );
    assert( connect(ui->showParticlesCheckbox, SIGNAL(toggled(bool)), ui->showParticlesCombo, SLOT(setEnabled(bool))) );
    assert( connect(ui->viewPanel, SIGNAL(showParticles()), ui->showParticlesCheckbox, SLOT(click())) );
    assert( connect(ui->viewPanel, SIGNAL(showMeshes()), ui->showContainersCheckbox, SLOT(click())) );
    CheckboxBoolAttribute::bindBool( ui->showContainersCheckbox, &UiSettings::showContainers(), this );
    ComboIntAttribute::bindInt( ui->showContainersCombo, &UiSettings::showContainersMode(), this );
    CheckboxBoolAttribute::bindBool( ui->showCollidersCheckbox, &UiSettings::showColliders(), this );
    ComboIntAttribute::bindInt( ui->showCollidersCombo, &UiSettings::showCollidersMode(), this );
    CheckboxBoolAttribute::bindBool( ui->showGridCheckbox, &UiSettings::showGrid(), this );
    ComboIntAttribute::bindInt( ui->showGridCombo, &UiSettings::showGridMode(), this );
    CheckboxBoolAttribute::bindBool( ui->showGridDataCheckbox, &UiSettings::showGridData(), this );
    ComboIntAttribute::bindInt( ui->showGridDataCombo, &UiSettings::showGridDataMode(), this );
    CheckboxBoolAttribute::bindBool( ui->showParticlesCheckbox, &UiSettings::showParticles(), this );
    ComboIntAttribute::bindInt( ui->showParticlesCombo, &UiSettings::showParticlesMode(), this );

    // Tools
    ui->toolButtonGroup->setId( ui->selectionToolButton, Tool::SELECTION );
    ui->toolButtonGroup->setId( ui->moveToolButton, Tool::MOVE );
    ui->toolButtonGroup->setId( ui->rotateToolButton, Tool::ROTATE );
    ui->toolButtonGroup->setId( ui->scaleToolButton, Tool::SCALE );
    ui->toolButtonGroup->addButton(ui->velocityToolButton);
    ui->toolButtonGroup->setId( ui->velocityToolButton, Tool::VELOCITY);
    assert( connect(ui->toolButtonGroup, SIGNAL(buttonClicked(int)), ui->viewPanel, SLOT(setTool(int))) );
    ui->selectionToolButton->click();

    // Selected Object
//    assert( connect(ui->viewPanel,SIGNAL(changeVelMag(float,bool)),this,SLOT(setVelMagText(float,bool))) );
//    assert( connect(ui->viewPanel,SIGNAL(changeVelVec(vec3,bool)),this,SLOT(setVelVecText(vec3,bool))) );
    assert( connect(ui->viewPanel,SIGNAL(changeVel(bool,float,float,float,float)),this,SLOT(setVelText(bool,float,float,float,float))) );
    assert( connect(ui->viewPanel,SIGNAL(changeSelection(QString,bool,int)),this,SLOT(setSelectionText(QString,bool,int))) );


    ui->toolGroupBox->init();
    ui->SelectedObjectGroupBox->init();
    ui->snowContainersGroupBox->init();
    ui->simulationGroupBox->init();
    ui->gridGroupBox->init();
    ui->exportGroupBox->init();
    ui->parametersGroupBox->init();
    ui->collidersGroupBox->init();
    ui->viewPanelGroupBox->init();
}

void MainWindow::keyPressEvent( QKeyEvent *event )
{
    if ( event->key() == Qt::Key_Q ) {
        ui->selectionToolButton->click();
        event->accept();
    } else if ( event->key() == Qt::Key_W ) {
        ui->moveToolButton->click();
        event->accept();
    } else if ( event->key() == Qt::Key_E ) {
        ui->rotateToolButton->click();
        event->accept();
    } else if ( event->key() == Qt::Key_R ) {
        ui->scaleToolButton->click();
        event->accept();
    } else {
        event->setAccepted( false );
    }
}

void MainWindow::resizeEvent( QResizeEvent* )
{
    UiSettings::windowSize() = size();
}

void MainWindow::moveEvent( QMoveEvent* )
{
    UiSettings::windowPosition() = pos();
}
