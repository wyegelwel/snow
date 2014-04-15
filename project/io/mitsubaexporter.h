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

#include "geometry/bbox.h"
#include "scene/scenenode.h"
#include "sim/particle.h"

//class MitsubaRenderSettings
//{
//    // TODO - populate with general render settings
//};

/**
 * @brief The MitsubaExporter class exports the scene
 * to a Mitsuba-readable XML format.
 * The snow is rendered to a grid-based volume data source (gridvolume)
 * our colliders are basic geometric primitives
 * and mitsuba can render them once we write them out.
 */


class MitsubaExporter
{
public:
    MitsubaExporter();

    /**
     * @brief exports particle system at a single time frame to a .vol file
     * to be rendered as a heterogenous medium in the Mitsuba renderer.
     * the format is unique to Wenzel Jakob's fsolve program
     * http://www.mitsuba-renderer.org/misc.html#
     * bounds specifies the maximum bounds of the heterogenous volume. smaller the better
     */
    static void exportPointCloud(QVector<Particle> particles, BBox bounds);

    /**
     * @brief exports scene at a particular time frame to a MITSUBA-renderable file format.
     * writes sceneNode and all of its children (i.e. pass in m_scene->root() to render the whole scene)
     */
    static void exportScene(std::string dir, int frame, SceneNode * sceneNode);


};

#endif // MITSUBAEXPORTER_H
