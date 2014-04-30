/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mitsubaexporter.h
**   Author: evjang, mliberma, taparson, wyegelwe
**   Created: 13 Apr 2014
**
**************************************************************************/

#ifndef MITSUBAEXPORTER_H
#define MITSUBAEXPORTER_H

#include <QString>
#include <QtXml>
#include <QtConcurrentRun>
#include <glm/geometric.hpp>
#include "geometry/grid.h"
#include "geometry/bbox.h"
#include "sim/particlegridnode.h"

class ImplicitCollider;
class SceneNode;

class MitsubaExporter
{
public:
    MitsubaExporter();
    MitsubaExporter(QString fprefix, int fps);
    ~MitsubaExporter();

    /**
     * @brief exports scene at a particular time frame to a MITSUBA-renderable file format.
     * writes sceneNode and all of its children (i.e. pass in m_scene->root() to render the whole scene)
     * it also needs access to the camera, so we also need to pass in the cam
     */

    float getspf();
    float getLastUpdateTime();
    void reset(Grid grid);
    Node * getNodesPtr();
    void runExportThread(float t);
    void exportScene(float t);

private:
    /**
     * @brief exports particle system at a single time frame to a .vol file
     * to be rendered as a heterogenous medium in the Mitsuba renderer.
     * the format is unique to Wenzel Jakob's fsolve program
     * http://www.mitsuba-renderer.org/misc.html#
     * bounds specifies the maximum bounds of the heterogenous volume. smaller the better
     */
    void exportDensityData(float t);
    void exportVelocityData(float t);
    void init();

    QString m_fileprefix;
    float m_lastUpdateTime;
    int m_fps; // number of frames to export every second of simulation
    float m_spf; // seconds per frame
    Node * m_nodes;
    Grid m_grid;
    int m_frame;
    bool m_busy;

    QFuture<void> m_future; // promise object used with QtConcurrentRun
};

#endif // MITSUBAEXPORTER_H
