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

class Engine;
class Camera;
class BBox;
class Particle;
class ParticleGrid;
class SceneNode;

//class MitsubaRenderSettings
//{
//    // TODO - populate with general render settings
//};


class MitsubaExporter
{
public:
    MitsubaExporter();

    /**
     * @brief exports scene at a particular time frame to a MITSUBA-renderable file format.
     * writes sceneNode and all of its children (i.e. pass in m_scene->root() to render the whole scene)
     * it also needs access to the camera, so we also need to pass in the cam
     */
    void exportScene(QString fprefix, int frame, Engine * engine, Camera * camera);

private:
    /**
     * @brief exports particle system at a single time frame to a .vol file
     * to be rendered as a heterogenous medium in the Mitsuba renderer.
     * the format is unique to Wenzel Jakob's fsolve program
     * http://www.mitsuba-renderer.org/misc.html#
     * bounds specifies the maximum bounds of the heterogenous volume. smaller the better
     */
    void exportVolumeData(QString fprefix);

    /**
     * adds <medium> tag to XML tree. Also calls exportVolumeData to write out volume.
     * calls exportVolumeData, then if successful, links to the .vol file
     */
    QDomElement appendMedium(QDomElement node);

    /// export rendering presets
    QDomElement appendRenderer(QDomElement node);

    /// converts camera into XML node
    QDomElement appendCamera(QDomElement node, Camera * camera);

    /// append transform matrix
    QDomElement appendXform(QDomElement node, glm::mat4 xform);

    /// appends sceneNodes
    QDomElement appendShape(QDomElement node, SceneNode * sceneNode);

    /// add obj shape node
    QDomElement appendOBJ(QDomElement node, QString objfile);

    /// add default material
    QDomElement appendBSDF(QDomElement node);

    /// adds a light sphere to the scene
    QDomElement appendLight(QDomElement node);

    QDomDocument m_document;
    QString      m_filename;

};

#endif // MITSUBAEXPORTER_H
