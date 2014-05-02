/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mitsubaexporter.cpp
**   Author: evjang
**   Created: 13 Apr 2014
**
**************************************************************************/

#include "mitsubaexporter.h"
#include <QFile>
#include <iostream>
#include "scene/scenenode.h"
#include "geometry/bbox.h"
#include "scene/scene.h"
#include "sim/particle.h"
#include "sim/engine.h"
#include "viewport/camera.h"
#include "math.h"
#include <fstream>
#include <iomanip>
#include "common/common.h"
#include <stdio.h>
#include "ui/uisettings.h"

MitsubaExporter::MitsubaExporter()
{
    m_fps = 24.f;
    m_fileprefix = "./mts";
    init();
}

MitsubaExporter::MitsubaExporter( QString fprefix, int fps )
  : m_fps(fps)
{
    // this has been temporarily hardcoded instead of prompting user for a second export prefix.
    // this is the only directory big enough to handle it anyway
    //m_fileprefix = QString("%1%2").arg("/data/people/evjang/offline_renders/", QFileInfo(fprefix).baseName());
    m_fileprefix = fprefix;
    init();
}

void MitsubaExporter::init()
{
    m_busy = false;
    m_nodes = NULL;
    m_spf = 1.f/float(m_fps);
    m_lastUpdateTime = 0.f;
    m_frame = 0;
}

MitsubaExporter::~MitsubaExporter()
{
    SAFE_DELETE_ARRAY(m_nodes);
}

float MitsubaExporter::getspf() {return m_spf;}
float MitsubaExporter::getLastUpdateTime() {return m_lastUpdateTime;}

void MitsubaExporter::reset(Grid grid)
{
    SAFE_DELETE_ARRAY(m_nodes);
    m_grid = grid;
    m_nodes = new Node[m_grid.nodeCount()];
}

void MitsubaExporter::runExportThread(float t)
{
    if (m_busy)
        m_future.waitForFinished();
    m_future = QtConcurrent::run(this, &MitsubaExporter::exportScene, t);
}

void MitsubaExporter::exportScene(float t)
{
    m_busy = true;
    // do work here
    if (UiSettings::exportDensity())
        exportDensityData(t);
    if (UiSettings::exportVelocity())
        exportVelocityData(t);
    // colliders are written to the scenefile from SceneIO because they only write once
    m_lastUpdateTime = t;
    m_frame += 1;
    m_busy = false;
}

void MitsubaExporter::writeVOLHeader(std::ofstream &os, const int channels)
{
    // header for VOL data

    int xres,yres,zres;
    xres = m_grid.nodeDim().x;
    yres = m_grid.nodeDim().y;
    zres = m_grid.nodeDim().z;
    const float h = m_grid.h;

    os.write("VOL", 3);
    char version = 3;
    os.write((char *) &version, sizeof(char));
    int value = 1;
    os.write((char *) &value, sizeof(int)); //Dense float32-based representation
    os.write((char *) &xres, sizeof(int));
    os.write((char *) &yres, sizeof(int));
    os.write((char *) &zres, sizeof(int));
    os.write((char *) &channels, sizeof(int));

    // the bounding box corresponds exactly where the heterogenous medium
    // will be positioned in MitexportVolsuba scene world space. If box is not
    // same size, stretching will occur. This is annoying when setting
    // up for arbitrary scenes, so the blender plugin will support re-writing these values
    // before rendering.

    float minX = m_grid.pos.x;
    float minY = m_grid.pos.y;
    float minZ = m_grid.pos.z;
    float maxX = minX + h*m_grid.dim.x;
    float maxY = minY + h*m_grid.dim.y;
    float maxZ = minZ + h*m_grid.dim.z;

    // bounding box
    os.write((char *) &minX, sizeof(float));
    os.write((char *) &minY, sizeof(float));
    os.write((char *) &minZ, sizeof(float));
    os.write((char *) &maxX, sizeof(float));
    os.write((char *) &maxY, sizeof(float));
    os.write((char *) &maxZ, sizeof(float));
}

void MitsubaExporter::exportDensityData(float t)
{
    QString fname = QString("%1_D_%2.vol").arg(m_fileprefix, QString("%1").arg(m_frame,4,'d',0,'0'));
    std::ofstream os(fname.toStdString().c_str());

    writeVOLHeader(os, 1);

    int xres,yres,zres;
    xres = m_grid.nodeDim().x;
    yres = m_grid.nodeDim().y;
    zres = m_grid.nodeDim().z;

    float h = m_grid.h;
    float v = h*h*h;

    for ( int k = 0; k < zres; ++k ) {
        for ( int j = 0; j < yres; ++j ) {
            for ( int i = 0; i < xres; ++i ) {
                int gIndex = (i*yres + j)*zres + k;
                float density = m_nodes[gIndex].mass / v;
                density *= 10000;                    // TODO, fix this when we have more particles.
                density = std::min(1.f,density);
                os.write((char *) &density, sizeof(float));
            }
        }
    }
    os.close();
}

void MitsubaExporter::exportVelocityData(float t)
{
    QString fname = QString("%1_V_%2.vol").arg(m_fileprefix, QString("%1").arg(m_frame,4,'d',0,'0'));
    std::ofstream os(fname.toStdString().c_str());

    writeVOLHeader(os, 3);

    int xres,yres,zres;
    xres = m_grid.nodeDim().x;
    yres = m_grid.nodeDim().y;
    zres = m_grid.nodeDim().z;

    float h = m_grid.h;
    float v = h*h*h;

    for ( int k = 0; k < zres; ++k ) {
        for ( int j = 0; j < yres; ++j ) {
            for ( int i = 0; i < xres; ++i ) {
                int gIndex = (i*yres + j)*zres + k;
                vec3 velocity = vec3::min(vec3(1), vec3::abs(m_nodes[gIndex].velocity));
                for (int c=0; c < 3; ++c) // RGB color channels
                {
                    os.write((char *) &velocity[c], sizeof(float));
                }
            }
        }
    }
    os.close();
}

Node * MitsubaExporter::getNodesPtr()
{
    return m_nodes;
}
