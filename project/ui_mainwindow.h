/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created: Wed Apr 30 23:39:13 2014
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QFrame>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QPushButton>
#include <QtGui/QScrollArea>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QStatusBar>
#include <QtGui/QToolBar>
#include <QtGui/QWidget>
#include "ui/collapsiblebox.h"
#include "ui/viewpanel.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionScreenshot;
    QAction *actionDemoTeapot;
    QAction *actionSave_Mesh;
    QAction *actionLoad_Scene;
    QAction *actionSave_Scene;
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    ViewPanel *viewPanel;
    QScrollArea *controlPanel;
    QWidget *scrollAreaWidgetContents;
    QGridLayout *gridLayout_2;
    CollapsibleBox *toolGroupBox;
    QHBoxLayout *horizontalLayout;
    QPushButton *selectionToolButton;
    QPushButton *moveToolButton;
    QPushButton *rotateToolButton;
    QPushButton *scaleToolButton;
    QSpacerItem *horizontalSpacer;
    QFrame *line;
    QSpacerItem *verticalSpacer;
    QFrame *line_5;
    CollapsibleBox *collidersGroupBox;
    QGridLayout *gridLayout_6;
    QComboBox *chooseCollider;
    QPushButton *colliderAddButton;
    CollapsibleBox *simulationGroupBox;
    QGridLayout *gridLayout_4;
    QPushButton *pauseButton;
    CollapsibleBox *parametersGroupBox;
    QGridLayout *gridLayout_7;
    QDoubleSpinBox *timeStepSpinbox;
    QLabel *timeStepLabel;
    QPushButton *startButton;
    QPushButton *resetButton;
    CollapsibleBox *exportGroupBox;
    QGridLayout *exportBox;
    QSpinBox *exportFPSSpinBox;
    QLabel *maxTimeLabel;
    QLabel *fpsLabel;
    QDoubleSpinBox *maxTimeSpinBox;
    QCheckBox *exportDensityCheckbox;
    QCheckBox *exportVelocityCheckbox;
    QPushButton *stopButton;
    CollapsibleBox *gridGroupBox;
    QGridLayout *gridLayout_8;
    QSpinBox *gridXSpinbox;
    QLabel *label_3;
    QSpinBox *gridYSpinbox;
    QLabel *label_2;
    QLabel *label;
    QSpinBox *gridZSpinbox;
    QHBoxLayout *horizontalLayout_2;
    QLabel *gridResolutionLable;
    QDoubleSpinBox *gridResolutionSpinbox;
    CollapsibleBox *viewPanelGroupBox;
    QGridLayout *gridLayout_5;
    QCheckBox *showParticlesCheckbox;
    QCheckBox *showContainersCheckbox;
    QCheckBox *showGridCheckbox;
    QCheckBox *showGridDataCheckbox;
    QCheckBox *showCollidersCheckbox;
    QComboBox *showContainersCombo;
    QComboBox *showGridCombo;
    QComboBox *showGridDataCombo;
    QComboBox *showParticlesCombo;
    QComboBox *showCollidersCombo;
    QFrame *line_3;
    CollapsibleBox *snowContainersGroupBox;
    QGridLayout *gridLayout_3;
    QLabel *materialLabel;
    QLabel *fillNumParticlesLabel;
    QDoubleSpinBox *fillResolutionSpinbox;
    QComboBox *snowMaterialCombo;
    QDoubleSpinBox *densitySpinbox;
    QSpinBox *fillNumParticlesSpinbox;
    QLabel *densityLabel;
    QPushButton *importButton;
    QLabel *fillResolutionLabel;
    QPushButton *fillButton;
    CollapsibleBox *gridGroupBox1;
    QGridLayout *gridLayout_9;
    QDoubleSpinBox *doubleSpinBox;
    QLabel *label_4;
    QFrame *line_4;
    QFrame *line_2;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuDemo;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;
    QButtonGroup *toolButtonGroup;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1012, 663);
        MainWindow->setFocusPolicy(Qt::StrongFocus);
        actionScreenshot = new QAction(MainWindow);
        actionScreenshot->setObjectName(QString::fromUtf8("actionScreenshot"));
        actionDemoTeapot = new QAction(MainWindow);
        actionDemoTeapot->setObjectName(QString::fromUtf8("actionDemoTeapot"));
        actionSave_Mesh = new QAction(MainWindow);
        actionSave_Mesh->setObjectName(QString::fromUtf8("actionSave_Mesh"));
        actionLoad_Scene = new QAction(MainWindow);
        actionLoad_Scene->setObjectName(QString::fromUtf8("actionLoad_Scene"));
        actionSave_Scene = new QAction(MainWindow);
        actionSave_Scene->setObjectName(QString::fromUtf8("actionSave_Scene"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(3);
        gridLayout->setContentsMargins(3, 3, 3, 3);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        viewPanel = new ViewPanel(centralWidget);
        viewPanel->setObjectName(QString::fromUtf8("viewPanel"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(viewPanel->sizePolicy().hasHeightForWidth());
        viewPanel->setSizePolicy(sizePolicy);
        viewPanel->setFocusPolicy(Qt::StrongFocus);
        viewPanel->setAutoFillBackground(true);

        gridLayout->addWidget(viewPanel, 0, 1, 1, 1);

        controlPanel = new QScrollArea(centralWidget);
        controlPanel->setObjectName(QString::fromUtf8("controlPanel"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(controlPanel->sizePolicy().hasHeightForWidth());
        controlPanel->setSizePolicy(sizePolicy1);
        controlPanel->setMinimumSize(QSize(300, 0));
        controlPanel->setMaximumSize(QSize(200, 16777215));
        QFont font;
        font.setPointSize(9);
        controlPanel->setFont(font);
        controlPanel->setAutoFillBackground(true);
        controlPanel->setFrameShape(QFrame::Box);
        controlPanel->setFrameShadow(QFrame::Sunken);
        controlPanel->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, -467, 280, 1054));
        gridLayout_2 = new QGridLayout(scrollAreaWidgetContents);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        toolGroupBox = new CollapsibleBox(scrollAreaWidgetContents);
        toolGroupBox->setObjectName(QString::fromUtf8("toolGroupBox"));
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(toolGroupBox->sizePolicy().hasHeightForWidth());
        toolGroupBox->setSizePolicy(sizePolicy2);
        toolGroupBox->setAutoFillBackground(true);
        horizontalLayout = new QHBoxLayout(toolGroupBox);
        horizontalLayout->setSpacing(4);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(4, 9, 4, 9);
        selectionToolButton = new QPushButton(toolGroupBox);
        toolButtonGroup = new QButtonGroup(MainWindow);
        toolButtonGroup->setObjectName(QString::fromUtf8("toolButtonGroup"));
        toolButtonGroup->addButton(selectionToolButton);
        selectionToolButton->setObjectName(QString::fromUtf8("selectionToolButton"));
        QSizePolicy sizePolicy3(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(selectionToolButton->sizePolicy().hasHeightForWidth());
        selectionToolButton->setSizePolicy(sizePolicy3);
        selectionToolButton->setMinimumSize(QSize(40, 40));
        selectionToolButton->setMaximumSize(QSize(40, 40));
        selectionToolButton->setAutoFillBackground(true);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/cursor.png"), QSize(), QIcon::Normal, QIcon::Off);
        selectionToolButton->setIcon(icon);
        selectionToolButton->setIconSize(QSize(25, 25));
        selectionToolButton->setCheckable(true);
        selectionToolButton->setChecked(false);

        horizontalLayout->addWidget(selectionToolButton);

        moveToolButton = new QPushButton(toolGroupBox);
        toolButtonGroup->addButton(moveToolButton);
        moveToolButton->setObjectName(QString::fromUtf8("moveToolButton"));
        sizePolicy3.setHeightForWidth(moveToolButton->sizePolicy().hasHeightForWidth());
        moveToolButton->setSizePolicy(sizePolicy3);
        moveToolButton->setMinimumSize(QSize(40, 40));
        moveToolButton->setMaximumSize(QSize(40, 40));
        moveToolButton->setAutoFillBackground(true);
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/move.png"), QSize(), QIcon::Normal, QIcon::Off);
        moveToolButton->setIcon(icon1);
        moveToolButton->setIconSize(QSize(30, 30));
        moveToolButton->setCheckable(true);
        moveToolButton->setChecked(false);

        horizontalLayout->addWidget(moveToolButton);

        rotateToolButton = new QPushButton(toolGroupBox);
        toolButtonGroup->addButton(rotateToolButton);
        rotateToolButton->setObjectName(QString::fromUtf8("rotateToolButton"));
        sizePolicy3.setHeightForWidth(rotateToolButton->sizePolicy().hasHeightForWidth());
        rotateToolButton->setSizePolicy(sizePolicy3);
        rotateToolButton->setMinimumSize(QSize(40, 40));
        rotateToolButton->setMaximumSize(QSize(40, 40));
        rotateToolButton->setAutoFillBackground(true);
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/rotate.png"), QSize(), QIcon::Normal, QIcon::Off);
        rotateToolButton->setIcon(icon2);
        rotateToolButton->setIconSize(QSize(30, 30));
        rotateToolButton->setCheckable(true);
        rotateToolButton->setChecked(false);

        horizontalLayout->addWidget(rotateToolButton);

        scaleToolButton = new QPushButton(toolGroupBox);
        toolButtonGroup->addButton(scaleToolButton);
        scaleToolButton->setObjectName(QString::fromUtf8("scaleToolButton"));
        sizePolicy3.setHeightForWidth(scaleToolButton->sizePolicy().hasHeightForWidth());
        scaleToolButton->setSizePolicy(sizePolicy3);
        scaleToolButton->setMinimumSize(QSize(40, 40));
        scaleToolButton->setMaximumSize(QSize(40, 40));
        scaleToolButton->setAutoFillBackground(true);
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/icons/scale.png"), QSize(), QIcon::Normal, QIcon::Off);
        scaleToolButton->setIcon(icon3);
        scaleToolButton->setIconSize(QSize(30, 30));
        scaleToolButton->setCheckable(true);
        scaleToolButton->setChecked(false);

        horizontalLayout->addWidget(scaleToolButton);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        gridLayout_2->addWidget(toolGroupBox, 0, 0, 1, 1);

        line = new QFrame(scrollAreaWidgetContents);
        line->setObjectName(QString::fromUtf8("line"));
        line->setAutoFillBackground(true);
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(line, 3, 0, 1, 1);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer, 10, 0, 1, 1);

        line_5 = new QFrame(scrollAreaWidgetContents);
        line_5->setObjectName(QString::fromUtf8("line_5"));
        line_5->setAutoFillBackground(true);
        line_5->setFrameShape(QFrame::HLine);
        line_5->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(line_5, 1, 0, 1, 1);

        collidersGroupBox = new CollapsibleBox(scrollAreaWidgetContents);
        collidersGroupBox->setObjectName(QString::fromUtf8("collidersGroupBox"));
        collidersGroupBox->setAutoFillBackground(true);
        gridLayout_6 = new QGridLayout(collidersGroupBox);
        gridLayout_6->setSpacing(4);
        gridLayout_6->setContentsMargins(11, 11, 11, 11);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        gridLayout_6->setContentsMargins(4, 9, 4, 9);
        chooseCollider = new QComboBox(collidersGroupBox);
        chooseCollider->setObjectName(QString::fromUtf8("chooseCollider"));
        chooseCollider->setAutoFillBackground(true);

        gridLayout_6->addWidget(chooseCollider, 0, 0, 1, 1);

        colliderAddButton = new QPushButton(collidersGroupBox);
        colliderAddButton->setObjectName(QString::fromUtf8("colliderAddButton"));
        colliderAddButton->setAutoFillBackground(true);

        gridLayout_6->addWidget(colliderAddButton, 0, 1, 1, 1);


        gridLayout_2->addWidget(collidersGroupBox, 6, 0, 1, 1);

        simulationGroupBox = new CollapsibleBox(scrollAreaWidgetContents);
        simulationGroupBox->setObjectName(QString::fromUtf8("simulationGroupBox"));
        simulationGroupBox->setAutoFillBackground(true);
        gridLayout_4 = new QGridLayout(simulationGroupBox);
        gridLayout_4->setSpacing(4);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        gridLayout_4->setContentsMargins(4, 9, 4, 9);
        pauseButton = new QPushButton(simulationGroupBox);
        pauseButton->setObjectName(QString::fromUtf8("pauseButton"));
        pauseButton->setEnabled(false);
        pauseButton->setAutoFillBackground(true);
        pauseButton->setCheckable(true);

        gridLayout_4->addWidget(pauseButton, 0, 1, 1, 1);

        parametersGroupBox = new CollapsibleBox(simulationGroupBox);
        parametersGroupBox->setObjectName(QString::fromUtf8("parametersGroupBox"));
        parametersGroupBox->setAutoFillBackground(true);
        gridLayout_7 = new QGridLayout(parametersGroupBox);
        gridLayout_7->setSpacing(6);
        gridLayout_7->setContentsMargins(4, 4, 4, 4);
        gridLayout_7->setObjectName(QString::fromUtf8("gridLayout_7"));
        gridLayout_7->setHorizontalSpacing(10);
        gridLayout_7->setVerticalSpacing(4);
        timeStepSpinbox = new QDoubleSpinBox(parametersGroupBox);
        timeStepSpinbox->setObjectName(QString::fromUtf8("timeStepSpinbox"));
        sizePolicy2.setHeightForWidth(timeStepSpinbox->sizePolicy().hasHeightForWidth());
        timeStepSpinbox->setSizePolicy(sizePolicy2);
        timeStepSpinbox->setAutoFillBackground(true);
        timeStepSpinbox->setDecimals(5);
        timeStepSpinbox->setMinimum(1e-05);
        timeStepSpinbox->setMaximum(0.1);
        timeStepSpinbox->setSingleStep(1e-05);
        timeStepSpinbox->setValue(1e-05);

        gridLayout_7->addWidget(timeStepSpinbox, 0, 1, 1, 1);

        timeStepLabel = new QLabel(parametersGroupBox);
        timeStepLabel->setObjectName(QString::fromUtf8("timeStepLabel"));
        sizePolicy2.setHeightForWidth(timeStepLabel->sizePolicy().hasHeightForWidth());
        timeStepLabel->setSizePolicy(sizePolicy2);
        timeStepLabel->setAutoFillBackground(true);
        timeStepLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_7->addWidget(timeStepLabel, 0, 0, 1, 1);


        gridLayout_4->addWidget(parametersGroupBox, 3, 0, 1, 2);

        startButton = new QPushButton(simulationGroupBox);
        startButton->setObjectName(QString::fromUtf8("startButton"));
        startButton->setAutoFillBackground(true);

        gridLayout_4->addWidget(startButton, 0, 0, 1, 1);

        resetButton = new QPushButton(simulationGroupBox);
        resetButton->setObjectName(QString::fromUtf8("resetButton"));
        resetButton->setAutoFillBackground(true);

        gridLayout_4->addWidget(resetButton, 1, 1, 1, 1);

        exportGroupBox = new CollapsibleBox(simulationGroupBox);
        exportGroupBox->setObjectName(QString::fromUtf8("exportGroupBox"));
        exportGroupBox->setAutoFillBackground(true);
        exportGroupBox->setCheckable(false);
        exportBox = new QGridLayout(exportGroupBox);
        exportBox->setSpacing(4);
        exportBox->setContentsMargins(4, 4, 4, 4);
        exportBox->setObjectName(QString::fromUtf8("exportBox"));
        exportFPSSpinBox = new QSpinBox(exportGroupBox);
        exportFPSSpinBox->setObjectName(QString::fromUtf8("exportFPSSpinBox"));
        exportFPSSpinBox->setAutoFillBackground(true);

        exportBox->addWidget(exportFPSSpinBox, 5, 1, 1, 1);

        maxTimeLabel = new QLabel(exportGroupBox);
        maxTimeLabel->setObjectName(QString::fromUtf8("maxTimeLabel"));

        exportBox->addWidget(maxTimeLabel, 4, 0, 1, 1, Qt::AlignRight);

        fpsLabel = new QLabel(exportGroupBox);
        fpsLabel->setObjectName(QString::fromUtf8("fpsLabel"));
        fpsLabel->setAutoFillBackground(true);

        exportBox->addWidget(fpsLabel, 5, 0, 1, 1, Qt::AlignRight);

        maxTimeSpinBox = new QDoubleSpinBox(exportGroupBox);
        maxTimeSpinBox->setObjectName(QString::fromUtf8("maxTimeSpinBox"));

        exportBox->addWidget(maxTimeSpinBox, 4, 1, 1, 1);

        exportDensityCheckbox = new QCheckBox(exportGroupBox);
        exportDensityCheckbox->setObjectName(QString::fromUtf8("exportDensityCheckbox"));
        QFont font1;
        font1.setBold(false);
        font1.setWeight(50);
        exportDensityCheckbox->setFont(font1);
        exportDensityCheckbox->setLayoutDirection(Qt::RightToLeft);
        exportDensityCheckbox->setAutoFillBackground(true);

        exportBox->addWidget(exportDensityCheckbox, 0, 1, 1, 1);

        exportVelocityCheckbox = new QCheckBox(exportGroupBox);
        exportVelocityCheckbox->setObjectName(QString::fromUtf8("exportVelocityCheckbox"));
        exportVelocityCheckbox->setFont(font1);
        exportVelocityCheckbox->setLayoutDirection(Qt::RightToLeft);

        exportBox->addWidget(exportVelocityCheckbox, 1, 1, 1, 1);


        gridLayout_4->addWidget(exportGroupBox, 5, 0, 1, 2);

        stopButton = new QPushButton(simulationGroupBox);
        stopButton->setObjectName(QString::fromUtf8("stopButton"));
        stopButton->setEnabled(false);
        stopButton->setAutoFillBackground(true);

        gridLayout_4->addWidget(stopButton, 1, 0, 1, 1);

        gridGroupBox = new CollapsibleBox(simulationGroupBox);
        gridGroupBox->setObjectName(QString::fromUtf8("gridGroupBox"));
        gridGroupBox->setAutoFillBackground(true);
        gridLayout_8 = new QGridLayout(gridGroupBox);
        gridLayout_8->setSpacing(4);
        gridLayout_8->setContentsMargins(4, 4, 4, 4);
        gridLayout_8->setObjectName(QString::fromUtf8("gridLayout_8"));
        gridXSpinbox = new QSpinBox(gridGroupBox);
        gridXSpinbox->setObjectName(QString::fromUtf8("gridXSpinbox"));
        gridXSpinbox->setAutoFillBackground(true);
        gridXSpinbox->setMinimum(1);
        gridXSpinbox->setMaximum(512);

        gridLayout_8->addWidget(gridXSpinbox, 0, 1, 1, 1);

        label_3 = new QLabel(gridGroupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        sizePolicy3.setHeightForWidth(label_3->sizePolicy().hasHeightForWidth());
        label_3->setSizePolicy(sizePolicy3);
        label_3->setAutoFillBackground(true);

        gridLayout_8->addWidget(label_3, 0, 4, 1, 1);

        gridYSpinbox = new QSpinBox(gridGroupBox);
        gridYSpinbox->setObjectName(QString::fromUtf8("gridYSpinbox"));
        gridYSpinbox->setAutoFillBackground(true);
        gridYSpinbox->setMinimum(1);
        gridYSpinbox->setMaximum(512);

        gridLayout_8->addWidget(gridYSpinbox, 0, 3, 1, 1);

        label_2 = new QLabel(gridGroupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        sizePolicy3.setHeightForWidth(label_2->sizePolicy().hasHeightForWidth());
        label_2->setSizePolicy(sizePolicy3);
        label_2->setAutoFillBackground(true);

        gridLayout_8->addWidget(label_2, 0, 2, 1, 1);

        label = new QLabel(gridGroupBox);
        label->setObjectName(QString::fromUtf8("label"));
        sizePolicy3.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy3);
        label->setAutoFillBackground(true);

        gridLayout_8->addWidget(label, 0, 0, 1, 1);

        gridZSpinbox = new QSpinBox(gridGroupBox);
        gridZSpinbox->setObjectName(QString::fromUtf8("gridZSpinbox"));
        gridZSpinbox->setAutoFillBackground(true);
        gridZSpinbox->setMinimum(1);
        gridZSpinbox->setMaximum(512);

        gridLayout_8->addWidget(gridZSpinbox, 0, 5, 1, 1);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        gridResolutionLable = new QLabel(gridGroupBox);
        gridResolutionLable->setObjectName(QString::fromUtf8("gridResolutionLable"));
        gridResolutionLable->setAutoFillBackground(true);
        gridResolutionLable->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_2->addWidget(gridResolutionLable);

        gridResolutionSpinbox = new QDoubleSpinBox(gridGroupBox);
        gridResolutionSpinbox->setObjectName(QString::fromUtf8("gridResolutionSpinbox"));
        gridResolutionSpinbox->setAutoFillBackground(true);
        gridResolutionSpinbox->setDecimals(3);
        gridResolutionSpinbox->setSingleStep(0.001);
        gridResolutionSpinbox->setValue(0.005);

        horizontalLayout_2->addWidget(gridResolutionSpinbox);


        gridLayout_8->addLayout(horizontalLayout_2, 1, 0, 1, 6);


        gridLayout_4->addWidget(gridGroupBox, 2, 0, 1, 2);


        gridLayout_2->addWidget(simulationGroupBox, 4, 0, 1, 1);

        viewPanelGroupBox = new CollapsibleBox(scrollAreaWidgetContents);
        viewPanelGroupBox->setObjectName(QString::fromUtf8("viewPanelGroupBox"));
        viewPanelGroupBox->setAutoFillBackground(true);
        gridLayout_5 = new QGridLayout(viewPanelGroupBox);
        gridLayout_5->setSpacing(4);
        gridLayout_5->setContentsMargins(11, 11, 11, 11);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        gridLayout_5->setContentsMargins(4, 9, 4, 9);
        showParticlesCheckbox = new QCheckBox(viewPanelGroupBox);
        showParticlesCheckbox->setObjectName(QString::fromUtf8("showParticlesCheckbox"));
        QFont font2;
        font2.setBold(true);
        font2.setWeight(75);
        showParticlesCheckbox->setFont(font2);
        showParticlesCheckbox->setLayoutDirection(Qt::RightToLeft);
        showParticlesCheckbox->setAutoFillBackground(true);

        gridLayout_5->addWidget(showParticlesCheckbox, 4, 0, 1, 1);

        showContainersCheckbox = new QCheckBox(viewPanelGroupBox);
        showContainersCheckbox->setObjectName(QString::fromUtf8("showContainersCheckbox"));
        showContainersCheckbox->setFont(font2);
        showContainersCheckbox->setLayoutDirection(Qt::RightToLeft);
        showContainersCheckbox->setAutoFillBackground(true);

        gridLayout_5->addWidget(showContainersCheckbox, 0, 0, 1, 1);

        showGridCheckbox = new QCheckBox(viewPanelGroupBox);
        showGridCheckbox->setObjectName(QString::fromUtf8("showGridCheckbox"));
        showGridCheckbox->setFont(font2);
        showGridCheckbox->setLayoutDirection(Qt::RightToLeft);
        showGridCheckbox->setAutoFillBackground(true);

        gridLayout_5->addWidget(showGridCheckbox, 2, 0, 1, 1);

        showGridDataCheckbox = new QCheckBox(viewPanelGroupBox);
        showGridDataCheckbox->setObjectName(QString::fromUtf8("showGridDataCheckbox"));
        showGridDataCheckbox->setFont(font2);
        showGridDataCheckbox->setLayoutDirection(Qt::RightToLeft);
        showGridDataCheckbox->setAutoFillBackground(true);

        gridLayout_5->addWidget(showGridDataCheckbox, 3, 0, 1, 1);

        showCollidersCheckbox = new QCheckBox(viewPanelGroupBox);
        showCollidersCheckbox->setObjectName(QString::fromUtf8("showCollidersCheckbox"));
        showCollidersCheckbox->setFont(font2);
        showCollidersCheckbox->setLayoutDirection(Qt::RightToLeft);

        gridLayout_5->addWidget(showCollidersCheckbox, 1, 0, 1, 1);

        showContainersCombo = new QComboBox(viewPanelGroupBox);
        showContainersCombo->setObjectName(QString::fromUtf8("showContainersCombo"));
        showContainersCombo->setAutoFillBackground(true);

        gridLayout_5->addWidget(showContainersCombo, 0, 1, 1, 1);

        showGridCombo = new QComboBox(viewPanelGroupBox);
        showGridCombo->setObjectName(QString::fromUtf8("showGridCombo"));
        showGridCombo->setAutoFillBackground(true);

        gridLayout_5->addWidget(showGridCombo, 2, 1, 1, 1);

        showGridDataCombo = new QComboBox(viewPanelGroupBox);
        showGridDataCombo->setObjectName(QString::fromUtf8("showGridDataCombo"));
        showGridDataCombo->setAutoFillBackground(true);

        gridLayout_5->addWidget(showGridDataCombo, 3, 1, 1, 1);

        showParticlesCombo = new QComboBox(viewPanelGroupBox);
        showParticlesCombo->setObjectName(QString::fromUtf8("showParticlesCombo"));
        showParticlesCombo->setAutoFillBackground(true);

        gridLayout_5->addWidget(showParticlesCombo, 4, 1, 1, 1);

        showCollidersCombo = new QComboBox(viewPanelGroupBox);
        showCollidersCombo->setObjectName(QString::fromUtf8("showCollidersCombo"));

        gridLayout_5->addWidget(showCollidersCombo, 1, 1, 1, 1);


        gridLayout_2->addWidget(viewPanelGroupBox, 8, 0, 1, 1);

        line_3 = new QFrame(scrollAreaWidgetContents);
        line_3->setObjectName(QString::fromUtf8("line_3"));
        line_3->setAutoFillBackground(true);
        line_3->setFrameShape(QFrame::HLine);
        line_3->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(line_3, 7, 0, 1, 1);

        snowContainersGroupBox = new CollapsibleBox(scrollAreaWidgetContents);
        snowContainersGroupBox->setObjectName(QString::fromUtf8("snowContainersGroupBox"));
        snowContainersGroupBox->setAutoFillBackground(true);
        gridLayout_3 = new QGridLayout(snowContainersGroupBox);
        gridLayout_3->setSpacing(4);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        gridLayout_3->setContentsMargins(4, 9, 4, 9);
        materialLabel = new QLabel(snowContainersGroupBox);
        materialLabel->setObjectName(QString::fromUtf8("materialLabel"));

        gridLayout_3->addWidget(materialLabel, 6, 0, 1, 1, Qt::AlignRight);

        fillNumParticlesLabel = new QLabel(snowContainersGroupBox);
        fillNumParticlesLabel->setObjectName(QString::fromUtf8("fillNumParticlesLabel"));
        fillNumParticlesLabel->setAutoFillBackground(true);
        fillNumParticlesLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_3->addWidget(fillNumParticlesLabel, 3, 0, 1, 1);

        fillResolutionSpinbox = new QDoubleSpinBox(snowContainersGroupBox);
        fillResolutionSpinbox->setObjectName(QString::fromUtf8("fillResolutionSpinbox"));
        fillResolutionSpinbox->setAutoFillBackground(true);
        fillResolutionSpinbox->setDecimals(4);
        fillResolutionSpinbox->setMinimum(0.0001);
        fillResolutionSpinbox->setMaximum(1);
        fillResolutionSpinbox->setSingleStep(0.0001);
        fillResolutionSpinbox->setValue(0.001);

        gridLayout_3->addWidget(fillResolutionSpinbox, 2, 1, 1, 1);

        snowMaterialCombo = new QComboBox(snowContainersGroupBox);
        snowMaterialCombo->setObjectName(QString::fromUtf8("snowMaterialCombo"));

        gridLayout_3->addWidget(snowMaterialCombo, 6, 1, 1, 1);

        densitySpinbox = new QDoubleSpinBox(snowContainersGroupBox);
        densitySpinbox->setObjectName(QString::fromUtf8("densitySpinbox"));
        densitySpinbox->setDecimals(0);
        densitySpinbox->setMinimum(1);
        densitySpinbox->setMaximum(1000);
        densitySpinbox->setValue(150);

        gridLayout_3->addWidget(densitySpinbox, 5, 1, 1, 1);

        fillNumParticlesSpinbox = new QSpinBox(snowContainersGroupBox);
        fillNumParticlesSpinbox->setObjectName(QString::fromUtf8("fillNumParticlesSpinbox"));
        fillNumParticlesSpinbox->setAutoFillBackground(true);
        fillNumParticlesSpinbox->setMinimum(0);
        fillNumParticlesSpinbox->setMaximum(524288);
        fillNumParticlesSpinbox->setSingleStep(512);
        fillNumParticlesSpinbox->setValue(65536);

        gridLayout_3->addWidget(fillNumParticlesSpinbox, 3, 1, 1, 1);

        densityLabel = new QLabel(snowContainersGroupBox);
        densityLabel->setObjectName(QString::fromUtf8("densityLabel"));
        densityLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_3->addWidget(densityLabel, 5, 0, 1, 1);

        importButton = new QPushButton(snowContainersGroupBox);
        importButton->setObjectName(QString::fromUtf8("importButton"));
        QSizePolicy sizePolicy4(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(importButton->sizePolicy().hasHeightForWidth());
        importButton->setSizePolicy(sizePolicy4);
        importButton->setMinimumSize(QSize(50, 0));
        importButton->setAutoFillBackground(true);

        gridLayout_3->addWidget(importButton, 1, 0, 1, 1);

        fillResolutionLabel = new QLabel(snowContainersGroupBox);
        fillResolutionLabel->setObjectName(QString::fromUtf8("fillResolutionLabel"));
        fillResolutionLabel->setAutoFillBackground(true);
        fillResolutionLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_3->addWidget(fillResolutionLabel, 2, 0, 1, 1);

        fillButton = new QPushButton(snowContainersGroupBox);
        fillButton->setObjectName(QString::fromUtf8("fillButton"));
        sizePolicy4.setHeightForWidth(fillButton->sizePolicy().hasHeightForWidth());
        fillButton->setSizePolicy(sizePolicy4);
        fillButton->setMinimumSize(QSize(50, 0));
        fillButton->setAutoFillBackground(true);

        gridLayout_3->addWidget(fillButton, 1, 1, 1, 1);

        gridGroupBox1 = new CollapsibleBox(snowContainersGroupBox);
        gridGroupBox1->setObjectName(QString::fromUtf8("gridGroupBox1"));
        gridLayout_9 = new QGridLayout(gridGroupBox1);
        gridLayout_9->setSpacing(6);
        gridLayout_9->setContentsMargins(11, 11, 11, 11);
        gridLayout_9->setObjectName(QString::fromUtf8("gridLayout_9"));
        doubleSpinBox = new QDoubleSpinBox(gridGroupBox1);
        doubleSpinBox->setObjectName(QString::fromUtf8("doubleSpinBox"));

        gridLayout_9->addWidget(doubleSpinBox, 0, 1, 1, 1);

        label_4 = new QLabel(gridGroupBox1);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_9->addWidget(label_4, 0, 0, 1, 1);


        gridLayout_3->addWidget(gridGroupBox1, 7, 0, 1, 2);


        gridLayout_2->addWidget(snowContainersGroupBox, 2, 0, 1, 1);

        line_4 = new QFrame(scrollAreaWidgetContents);
        line_4->setObjectName(QString::fromUtf8("line_4"));
        line_4->setAutoFillBackground(true);
        line_4->setFrameShape(QFrame::HLine);
        line_4->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(line_4, 9, 0, 1, 1);

        line_2 = new QFrame(scrollAreaWidgetContents);
        line_2->setObjectName(QString::fromUtf8("line_2"));
        line_2->setAutoFillBackground(true);
        line_2->setFrameShape(QFrame::HLine);
        line_2->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(line_2, 5, 0, 1, 1);

        controlPanel->setWidget(scrollAreaWidgetContents);

        gridLayout->addWidget(controlPanel, 0, 0, 1, 1);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1012, 29));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        menuDemo = new QMenu(menuBar);
        menuDemo->setObjectName(QString::fromUtf8("menuDemo"));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuDemo->menuAction());
        menuFile->addAction(actionScreenshot);
        menuFile->addAction(actionSave_Mesh);
        menuFile->addAction(actionLoad_Scene);
        menuFile->addAction(actionSave_Scene);
        menuDemo->addAction(actionDemoTeapot);

        retranslateUi(MainWindow);
        QObject::connect(actionScreenshot, SIGNAL(triggered()), MainWindow, SLOT(takeScreenshot()));
        QObject::connect(actionDemoTeapot, SIGNAL(triggered()), viewPanel, SLOT(teapotDemo()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0, QApplication::UnicodeUTF8));
        actionScreenshot->setText(QApplication::translate("MainWindow", "Take Screenshot", 0, QApplication::UnicodeUTF8));
        actionScreenshot->setShortcut(QApplication::translate("MainWindow", "Ctrl+P", 0, QApplication::UnicodeUTF8));
        actionDemoTeapot->setText(QApplication::translate("MainWindow", "Teapot", 0, QApplication::UnicodeUTF8));
        actionSave_Mesh->setText(QApplication::translate("MainWindow", "Save Mesh", 0, QApplication::UnicodeUTF8));
        actionLoad_Scene->setText(QApplication::translate("MainWindow", "Load Scene", 0, QApplication::UnicodeUTF8));
        actionSave_Scene->setText(QApplication::translate("MainWindow", "Save Scene", 0, QApplication::UnicodeUTF8));
        toolGroupBox->setTitle(QApplication::translate("MainWindow", "Tools", 0, QApplication::UnicodeUTF8));
        selectionToolButton->setText(QString());
        moveToolButton->setText(QString());
        rotateToolButton->setText(QString());
        collidersGroupBox->setTitle(QApplication::translate("MainWindow", "Collider", 0, QApplication::UnicodeUTF8));
        chooseCollider->clear();
        chooseCollider->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "Sphere", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Half-Plane", 0, QApplication::UnicodeUTF8)
        );
        colliderAddButton->setText(QApplication::translate("MainWindow", "addCollider", 0, QApplication::UnicodeUTF8));
        simulationGroupBox->setTitle(QApplication::translate("MainWindow", "Simulation", 0, QApplication::UnicodeUTF8));
        pauseButton->setText(QApplication::translate("MainWindow", "Pause", 0, QApplication::UnicodeUTF8));
        parametersGroupBox->setTitle(QApplication::translate("MainWindow", "Parameters", 0, QApplication::UnicodeUTF8));
        timeStepSpinbox->setSuffix(QApplication::translate("MainWindow", " s", 0, QApplication::UnicodeUTF8));
        timeStepLabel->setText(QApplication::translate("MainWindow", "Time Step", 0, QApplication::UnicodeUTF8));
        startButton->setText(QApplication::translate("MainWindow", "Start", 0, QApplication::UnicodeUTF8));
        resetButton->setText(QApplication::translate("MainWindow", "Reset", 0, QApplication::UnicodeUTF8));
        exportGroupBox->setTitle(QApplication::translate("MainWindow", "Export", 0, QApplication::UnicodeUTF8));
        maxTimeLabel->setText(QApplication::translate("MainWindow", "Duration (sec)", 0, QApplication::UnicodeUTF8));
        fpsLabel->setText(QApplication::translate("MainWindow", "FPS", 0, QApplication::UnicodeUTF8));
        exportDensityCheckbox->setText(QApplication::translate("MainWindow", "Grid Densities", 0, QApplication::UnicodeUTF8));
        exportVelocityCheckbox->setText(QApplication::translate("MainWindow", "Grid Velocities", 0, QApplication::UnicodeUTF8));
        stopButton->setText(QApplication::translate("MainWindow", "Stop", 0, QApplication::UnicodeUTF8));
        gridGroupBox->setTitle(QApplication::translate("MainWindow", "Grid", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("MainWindow", "Z", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("MainWindow", "Y", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("MainWindow", "X", 0, QApplication::UnicodeUTF8));
        gridResolutionLable->setText(QApplication::translate("MainWindow", "Cell Size", 0, QApplication::UnicodeUTF8));
        gridResolutionSpinbox->setSuffix(QApplication::translate("MainWindow", " m", 0, QApplication::UnicodeUTF8));
        viewPanelGroupBox->setTitle(QApplication::translate("MainWindow", "View Panel", 0, QApplication::UnicodeUTF8));
        showParticlesCheckbox->setText(QApplication::translate("MainWindow", "Particles", 0, QApplication::UnicodeUTF8));
        showContainersCheckbox->setText(QApplication::translate("MainWindow", "Containers", 0, QApplication::UnicodeUTF8));
        showGridCheckbox->setText(QApplication::translate("MainWindow", "Grid", 0, QApplication::UnicodeUTF8));
        showGridDataCheckbox->setText(QApplication::translate("MainWindow", "Grid Data", 0, QApplication::UnicodeUTF8));
        showCollidersCheckbox->setText(QApplication::translate("MainWindow", "Colliders", 0, QApplication::UnicodeUTF8));
        showContainersCombo->clear();
        showContainersCombo->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "Wireframe", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Solid", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Both", 0, QApplication::UnicodeUTF8)
        );
        showGridCombo->clear();
        showGridCombo->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "Box", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Min Face Cells", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "All Face Cells", 0, QApplication::UnicodeUTF8)
        );
        showGridDataCombo->clear();
        showGridDataCombo->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "Density", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Velocity", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Speed", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Force", 0, QApplication::UnicodeUTF8)
        );
        showParticlesCombo->clear();
        showParticlesCombo->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "Mass", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Velocity", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Speed", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Stiffness", 0, QApplication::UnicodeUTF8)
        );
        showCollidersCombo->clear();
        showCollidersCombo->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "Wireframe", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Solid", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Both", 0, QApplication::UnicodeUTF8)
        );
        snowContainersGroupBox->setTitle(QApplication::translate("MainWindow", "Snow Containers", 0, QApplication::UnicodeUTF8));
        materialLabel->setText(QApplication::translate("MainWindow", "Material Preset", 0, QApplication::UnicodeUTF8));
        fillNumParticlesLabel->setText(QApplication::translate("MainWindow", "Particles", 0, QApplication::UnicodeUTF8));
        fillResolutionSpinbox->setSuffix(QApplication::translate("MainWindow", " m", 0, QApplication::UnicodeUTF8));
        snowMaterialCombo->clear();
        snowMaterialCombo->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "Default", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MainWindow", "Chunky", 0, QApplication::UnicodeUTF8)
        );
        densitySpinbox->setSuffix(QApplication::translate("MainWindow", " kg/m3", 0, QApplication::UnicodeUTF8));
        densityLabel->setText(QApplication::translate("MainWindow", "Target Density", 0, QApplication::UnicodeUTF8));
        importButton->setText(QApplication::translate("MainWindow", "Import Mesh", 0, QApplication::UnicodeUTF8));
        fillResolutionLabel->setText(QApplication::translate("MainWindow", "Cell Size", 0, QApplication::UnicodeUTF8));
        fillButton->setText(QApplication::translate("MainWindow", "Fill Selected", 0, QApplication::UnicodeUTF8));
        gridGroupBox1->setTitle(QApplication::translate("MainWindow", "Material Parameters", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("MainWindow", "TextLabel", 0, QApplication::UnicodeUTF8));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", 0, QApplication::UnicodeUTF8));
        menuDemo->setTitle(QApplication::translate("MainWindow", "Demo", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
