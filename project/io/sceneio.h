/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   sceneparser.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#ifndef SCENEIO_H
#define SCENEIO_H

/**
 * @brief SceneIO class
 *
 * reads/writes scene files. Scene files contain not only static objects,
 * but the snow simulation parameters.
 *
 * Note, this is designed to be saved once when beginning a simulation with exporting
 * enabled, so this class was not designed with multithreading in mind.
 *
 */

#include <iostream>
#include <QString>
#include <QtXml>

class Scene;
class Engine;

class SceneIO
{
public:
    SceneIO();

    bool read(QString fname, Scene * scene, Engine * engine);
    bool write(QString fname, Scene * scene, Engine * engine);

    QString sceneFile() { return m_scenefile; }
    void setSceneFile(QString filename) { m_scenefile = filename; }
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
    //QDomElement appendLight(QDomElement node);

private:
    QString m_scenefile;
    QDomDocument m_document; // XML document

};

#endif // SCENEIO_H
