#-------------------------------------------------
#
# Project created by QtCreator 2014-04-03T18:12:01
#
#-------------------------------------------------

QT       += core
QT       -= gui

TARGET = CUDA_vectorAdd
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


HEADERS += \
    CUDA_helpers.h


SOURCES += main.cpp

# C++ flag
QMAKE_CXXFLAGS_RELEASE=-O3

# CUDA stuff
CUDA_SOURCES += kernel.cu

CUDA_DIR = /contrib/projects/cuda5-toolkit
INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64

LIBS += -lcudart -lcuda

OTHER_FILES += \
    kernel.cu

# GPU ARCH
# this gets passed as the gpu-architecture flag to nvcc compiler
# specifying particular architectures enable certain features, limited to the compute capability
# of the GPU. compute capabilities listed here http://en.wikipedia.org/wiki/CUDA
# MSLAB GeForce 460 seems to have compute capability 2.1
CUDA_ARCH = sm_21

# custom NVCC flags
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# compile CUDA kernels using nvcc
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
    2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o # suffix needed for this to work?
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda


