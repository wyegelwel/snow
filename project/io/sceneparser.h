/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   sceneparser.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#ifndef SCENEPARSER_H
#define SCENEPARSER_H

/**
 * @brief The SceneParser class
 *
 * reads scene files. Scene files contain not only static objects,
 * but the snow simulation parameters.
 *
 * In the future we may extend with support for moving objects.
 *
 */

#include <iostream>

class SceneNode;
class WorldParams;

class SceneParser
{
public:
    SceneParser();

    /**
     * returns a sceneNode corresponding to the parsed XML data
     * The simulation parameters will also read the world simulation params
     * (which contains details like lambda, mu, dt, etc)
     * Usage:
     *
     *
     * SceneNode * root;
     * WorldParams params;
     * SceneParser::read("snowballdrop.xml", root, params);
     *
     * // now add root, params to your scene.
     *
     */
    static void read(std::string fname, SceneNode * &node, WorldParams &params);

    /**
     * writes the data to XML format. Use case: we set up a simulation in the GUI
     * that we like and want to save it for offline rendering
     * (note, this scene is static and does not contain the snow particle data).
     *
     * That is up to the mitsubaexporter class and is done during rendering.
     *
     * SceneParser::write("myscene.xml", SceneNode * node);
     *
     */
    static void write();

};

#endif // SCENEPARSER_H
