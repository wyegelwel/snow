#-------------------------------------------------
#
# Project created by QtCreator 2014-04-06T18:10:31
#
#-------------------------------------------------

QT       += core gui opengl xml

# OpenGL stuff
LIBS += -lGLEW -lGLEWmx
DEFINES += GL_GLEXT_PROTOTYPES

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TEMPLATE = app
TARGET = snow

DEFINES += PROJECT_PATH=\\\"$$_PRO_FILE_PWD_\\\"

SOURCES += \
    main.cpp \
    ui/mainwindow.cpp \
    ui/viewpanel.cpp \
    ui/userinput.cpp \
    ui/infopanel.cpp \
    viewport/viewport.cpp \
    geometry/mesh.cpp \
    io/objparser.cpp \
    io/mitsubaexporter.cpp \
    scene/scene.cpp \
    scene/scenenode.cpp \
    tests/tests.cpp \
    sim/collider.cpp \
    geometry/bbox.cpp \
    sim/engine.cpp \
    io/sceneparser.cpp \
    ui/uisettings.cpp \
    ui/picker.cpp \
    ui/tools/selectiontool.cpp \
    ui/tools/movetool.cpp \
    scene/scenegrid.cpp \
    sim/particlesystem.cpp \
    sim/particlegrid.cpp \
    ui/tools/rotatetool.cpp \
    ui/tools/scaletool.cpp \
    ui/tools/tool.cpp \
    ui/collapsiblebox.cpp



HEADERS  += \
    ui/mainwindow.h \
    ui/viewpanel.h \
    ui/infopanel.h \
    ui/userinput.h \
    viewport/camera.h \
    viewport/viewport.h \
    common/common.h \
    sim/particle.h \
    cuda/functions.h \
    geometry/mesh.h \
    io/objparser.h \
    io/mitsubaexporter.h \
    scene/scene.h \
    scene/scenenode.h \
    common/renderable.h \
    common/types.h \
    tests/tests.h \
    sim/collider.h \
    geometry/bbox.h \
    common/math.h \
    geometry/grid.h \
    sim/engine.h \
    io/sceneparser.h \
    ui/databinding.h \
    ui/uisettings.h \
    sim/material.h \
    sim/parameters.h \
    sim/particlegridnode.h \
    ui/picker.h \
    scene/scenenodeiterator.h \
    ui/tools/tool.h \
    ui/tools/selectiontool.h \
    ui/tools/Tools.h \
    ui/tools/movetool.h \
    scene/scenegrid.h \
    sim/particlesystem.h \
    sim/particlegrid.h \
    ui/tools/rotatetool.h \
    ui/tools/scaletool.h \
    ui/collapsiblebox.h

FORMS    += ui/mainwindow.ui

# C++ flag
QMAKE_CXXFLAGS_RELEASE=-O3
QMAKE_CXXFLAGS += -std=c++11

# CUDA stuff
CUDA_SOURCES += cuda/snow.cu \
    cuda/mesh.cu \
#    cuda/wil.cu \
#    cuda/max.cu \
#    cuda/tim.cu \
#    cuda/eric.cu \
    cuda/simulation.cu

CUDA_DIR = /contrib/projects/cuda5-toolkit
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += $$CUDA_DIR/samples/common/inc
QMAKE_LIBDIR += $$CUDA_DIR/lib64

LIBS += -lcudart -lcuda

OTHER_FILES += \
    CUDA_notes.txt \
    cuda/snow.cu \
    cuda/mesh.cu \
    cuda/wil.cu \
    cuda/max.cu \
    cuda/tim.cu \
    cuda/eric.cu \
    cuda/decomposition.cu \
    cuda/weighting.cu \
    cuda/matrix.cu \
    cuda/vector.cu \
    cuda/quaternion.cu \
    cuda/simulation.cu \
    cuda/collider.cu \
    resources/shaders/particlesystem.vert \
    resources/shaders/particlesystem.frag \
    resources/shaders/particlegrid.frag \
    resources/shaders/particlegrid.vert

# GPU ARCH
# this gets passed as the gpu-architecture flag to nvcc compiler
# specifying particular architectures enable certain features, limited to the compute capability
# of the GPU. compute capabilities listed here http://en.wikipedia.org/wiki/CUDA
# MSLAB GeForce 460 seems to have compute capability 2.1
CUDA_ARCH = sm_21

# custom NVCC flags
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ') -I$$_PRO_FILE_PWD_

# compile CUDA kernels using nvcc
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
    2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o # suffix needed for this to work?
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

RESOURCES += \
    resources/icons/icons.qrc \
    resources/shaders/shaders.qrc
