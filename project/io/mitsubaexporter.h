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

class Scene;
class Camera;
class BBox;
class Particle;
class ParticleGrid;

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
     * @brief exports scene at a particular time frame to a MITSUBA-renderable file format.
     * writes sceneNode and all of its children (i.e. pass in m_scene->root() to render the whole scene)
     */
    void exportScene(QString fprefix, int frame, Scene * scene, Camera * camera);
private:
    /**
     * @brief exports particle system at a single time frame to a .vol file
     * to be rendered as a heterogenous medium in the Mitsuba renderer.
     * the format is unique to Wenzel Jakob's fsolve program
     * http://www.mitsuba-renderer.org/misc.html#
     * bounds specifies the maximum bounds of the heterogenous volume. smaller the better
     *
     * TODO - need to
     *
     */
    void exportVolumeData(QString fprefix, BBox bounds, ParticleGrid * grid);


    /**
     * adds <medium> tag to XML tree. Also calls exportVolumeData to write out volume.
     * calls exportVolumeData, then if successful, links to the .vol file
     */
    void addMedium(QDomElement &node);

    /**
     * exports the integrator presets
     */
    void addRenderer(QDomElement &node);

    /**
     * outputs camera into XML format for Mitsuba
     * TODO
     */
    void addCamera(QDomElement &node, Camera * camera);

    /**
     * adds generic obj node as child of node under doc.
     * used by addCollider and addSnowContainer
     */
    void addOBJ(QDomElement &node, QString objfile);

    /**
     * adds collider OBJ to XML tree
     */
    void addCollider(QDomElement &node);

    /**
     * adds snow container OBJ to XML tree
     */
    void addSnowContainer(QDomElement &node);


    QDomDocument m_document;
    QString      m_filename;

};

#endif // MITSUBAEXPORTER_H
