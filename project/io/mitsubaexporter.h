/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mitsubaexporter.h
**   Author: evjang
**   Created: 13 Apr 2014
**
**************************************************************************/

// generic class for exporting to Mitsuba for offline rendering


#ifndef MITSUBAEXPORTER_H
#define MITSUBAEXPORTER_H


#include <QString>
#include <QtXml>
#include <glm/geometric.hpp>
#include "geometry/grid.h"
#include "geometry/bbox.h"
#include "sim/particlegrid.h"

class SceneNode;


//class MitsubaRenderSettings
//{
//    // TODO - populate with general render settings
//};


class MitsubaExporter
{
public:
    MitsubaExporter();
    MitsubaExporter(QString fprefix, float fps);
    ~MitsubaExporter();

    /**
     * @brief exports scene at a particular time frame to a MITSUBA-renderable file format.
     * writes sceneNode and all of its children (i.e. pass in m_scene->root() to render the whole scene)
     * it also needs access to the camera, so we also need to pass in the cam
     */


    float getspf();
    float getLastUpdateTime();
    void reset(Grid grid);
    void exportVolumeData(float t);
    ParticleGrid::Node * getNodesPtr();

private:
    /**
     * @brief exports particle system at a single time frame to a .vol file
     * to be rendered as a heterogenous medium in the Mitsuba renderer.
     * the format is unique to Wenzel Jakob's fsolve program
     * http://www.mitsuba-renderer.org/misc.html#
     * bounds specifies the maximum bounds of the heterogenous volume. smaller the better
     */

    //void exportScene(QString fprefix, int frame); //, Engine * engine, Camera * camera

    /**
     * adds <medium> tag to XML tree. Also calls exportVolumeData to write out volume.
     * calls exportVolumeData, then if successful, links to the .vol file
//     */
//    QDomElement appendMedium(QDomElement node);

//    /// export rendering presets
//    QDomElement appendRenderer(QDomElement node);

//    /// converts camera into XML node
//    //QDomElement appendCamera(QDomElement node, Camera * camera);

//    /// append transform matrix
//    QDomElement appendXform(QDomElement node, glm::mat4 xform);

//    /// appends sceneNodes
//    QDomElement appendShape(QDomElement node, SceneNode * sceneNode);

//    /// add obj shape node
//    QDomElement appendOBJ(QDomElement node, QString objfile);

//    /// add default material
//    QDomElement appendBSDF(QDomElement node);

    /// adds a light sphere to the scene
    //QDomElement appendLight(QDomElement node);

    QDomDocument m_document;
    QString      m_fileprefix;

    float m_lastUpdateTime;
    float m_fps; // number of frames to export every second of simulation
    float m_spf; // seconds per frame

    // densities of each grid node
    //float * m_densities = NULL;

    ParticleGrid::Node * m_nodes;

    // scattering albedo of each grid node
    // http://en.wikipedia.org/wiki/Single-scattering_albedo
    //float * m_albedo = NULL;

    Grid m_grid;
    BBox m_bbox;

    int m_frame;
    bool m_busy;
};

#endif // MITSUBAEXPORTER_H
