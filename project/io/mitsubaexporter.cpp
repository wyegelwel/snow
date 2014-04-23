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
#include "sim/collider.h" // implicit colliders
#include "math.h"
#include <fstream>
#include <iomanip>
#include "common/common.h"
#include <stdio.h>

MitsubaExporter::MitsubaExporter()
{
    m_fps = 24.f;
    m_fileprefix = "./mts";
    init();
}

MitsubaExporter::MitsubaExporter(QString fprefix, int fps)
    : m_fileprefix(fprefix), m_fps(fps)
{
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
//    SAFE_DELETE_ARRAY(m_albedo);
    m_grid = grid;
    m_nodes = new ParticleGridNode[m_grid.nodeCount()];

    //m_albedo = new float[m_grid.nodeCount()];
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
    exportVolumeData(t);
//  exportColliders(m_colliders);
    m_lastUpdateTime = t;
    m_frame += 1;
    m_busy = false;
}

void MitsubaExporter::exportVolumeData(float t)
{
    QString fname = QString("%1_%2.vol").arg(m_fileprefix, QString("%1").arg(m_frame,4,'d',0,'0'));
    std::ofstream os(fname.toStdString().c_str());

    int xres,yres,zres;
    xres = m_grid.nodeDim().x;
    yres = m_grid.nodeDim().y;
    zres = m_grid.nodeDim().z;

    os.write("VOL", 3);
    char version = 3;
    os.write((char *) &version, sizeof(char));
    int value = 1;
    os.write((char *) &value, sizeof(int)); //Dense float32-based representation
    os.write((char *) &xres, sizeof(int));
    os.write((char *) &yres, sizeof(int));
    os.write((char *) &zres, sizeof(int));
    int channels = 1;      // number of channels
    os.write((char *) &channels, sizeof(int));

//    float minX = m_bbox.min().x;
//    float minY = m_bbox.min().y;
//    float minZ = m_bbox.min().z;
//    float maxX = m_bbox.max().x;
//    float maxY = m_bbox.max().y;
//    float maxZ = m_bbox.max().z;

    // the bounding box of the snow volume
    // corresponds to exactly where the heterogenous medium
    // will be positioned in the scene. irrelevant to everything else.
    // note, will warp if not proportional to simulation space

    float minX=-.5;
    float maxX=.5;
    float minY=0;
    float maxY=1;
    float minZ=-.5;
    float maxZ=.5;
    // bounding box
    os.write((char *) &minX, sizeof(float));
    os.write((char *) &minY, sizeof(float));
    os.write((char *) &minZ, sizeof(float));
    os.write((char *) &maxX, sizeof(float));
    os.write((char *) &maxY, sizeof(float));
    os.write((char *) &maxZ, sizeof(float));

    float h = m_grid.h;
    float v = h*h*h;

    for ( int k = 0, mIndex = 0; k < zres; ++k ) {
        for ( int j = 0; j < yres; ++j ) {
            for ( int i = 0; i < xres; ++i, ++mIndex ) {
                int gIndex = (i*yres + j)*zres + k;
                float density = m_nodes[gIndex].mass / v;
                density *= 10000;
                density = std::min(1.f,density);
                os.write((char *) &density, sizeof(float));
            }
        }
    }
    os.close();
}


ParticleGridNode * MitsubaExporter::getNodesPtr()
{
    return m_nodes;
}

/// temporarily put on hold - wait for the rest of the scenegraph implementation
/// to be frozen.

//void MitsubaExporter::exportScene(QString fprefix1, int frame) //, Engine *engine, Camera * camera
//{
    /**
     * Exports Single Frame of Scene to Mitsuba format.
     *
     * Traverses the scene graph and converts colliders to basic Mitsuba primitives
     * filled objects are loaded OBJs, and the snow is a heterogenous medium-type volume data
     *
     */
    //QString fname = fprefix + QString("_") + QString::number(frame,);
//    QString fprefix2 = QString("%1_%2").arg(fprefix1, QString("%1").arg(frame,4,'d',0,'0'));
//    exportVolumeData(fprefix2);
//    std::cout << "file written!" << std::endl;


//    // xml header
//    QDomProcessingInstruction pi = m_document.createProcessingInstruction("xml", "version=\"1.0\" encoding=\"utf-8\" ");
//    m_document.appendChild(pi);
//    // root element for the scene
//    QDomElement sceneNode = m_document.createElement("scene");
//    sceneNode.setAttribute("version", "0.5.0");
//    m_document.appendChild(sceneNode);

//    // we want a volumetric path tracer
//    appendRenderer(sceneNode);
//    // add the camera
//    appendCamera(sceneNode, camera);

//    // now traverse the scene graph for renderables.
//    // renderables are either snow containers
//    // or colliders.
//    Scene * scene = engine->scene();
//    QList<SceneNode *> nodes = scene->root()->allNodes();
//    for (int i=0;i<nodes.size();++i)
//    {
//        appendShape(sceneNode,nodes[i]);
//    }

//    // write it to file
//    QFile file(fname);
//    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
//    {
//        std::cout << "write failed" << std::endl;
//    }
//    else
//    {
//        QTextStream stream(&file);
//        stream << m_document.toString();
//        file.close();
//        std::cout << "file written" << std::endl;
//    }

//}


//QDomElement MitsubaExporter::appendCamera(QDomElement node, Camera *camera)
//{
//    //    <sensor type="perspective">
//    //		<float name="focusDistance" value="1.25668"/>
//    //		<float name="fov" value="45.8402"/>
//    //		<string name="fovAxis" value="x"/>
//    //		<transform name="toWorld">
//    //			<scale x="-1"/>

//    //			<lookat target="-0.166029, 0.148984, -0.537402" origin="-0.61423, 0.154197, -1.43132" up="-0.000640925, -0.999985, -0.0055102"/>
//    //		</transform>

//    //		<sampler type="ldsampler">
//    //			<integer name="sampleCount" value="64"/>
//    //		</sampler>

//    //		<film type="hdrfilm">
//    //			<integer name="height" value="576"/>
//    //			<integer name="width" value="768"/>

//    //			<rfilter type="gaussian"/>
//    //		</film>
//    //	</sensor>


//    //    QDomElement sensorNode = document.createElement("sensor");
//    //    sensorNode.setAttribute("type","perspective"); {
//    //        QDomElement fd = document.createElement("float");
//    //        fd.setAttribute("focusDistance", "1.25668");
//    //        sensorNode.appendChild(fd);

//    //        QDomElement fov = document.createElement("float");
//    //        fov.setAttribute("fov", "45.8402");
//    //        sensorNode.appendChild(fov);

//    //        QDomElement fovAxis = document.createElement("string");
//    //        fovAxis.setAttribute("fovAxis","x");
//    //        sensorNode.appendChild(fovAxis);

//            // transform node
//            // sampler node
//            // film node

//    //    }
//    //    sceneNode.appendChild(sensorNode);
//}

//QDomElement MitsubaExporter::appendLight(QDomElement node)
//{
//    // add some lights to the scene

////    <shape type="obj">
////		<!-- Shiny floor -->
////		<string name="filename" value="plane.obj"/>

////		<bsdf type="diffuse">
////			<rgb name="diffuseReflectance" value=".2, .2, .3"/>
////		</bsdf>
////		<transform name="toWorld">
////			<translate y=".48"/>
////		</transform>
////	</shape>


////    <shape type="sphere">
////		<point name="center" x="0" y="-2" z="-1"/>
////		<float name="radius" value=".2"/>

////		<emitter type="area">
////			<spectrum name="radiance" value="400"/>
////		</emitter>
////	</shape>

//}

//QDomElement MitsubaExporter::appendMedium(QDomElement node)
//{

//    //    <medium type="heterogeneous" id="smoke">
//    //		<string name="method" value="woodcock"/>

//    //		<volume name="density" type="gridvolume">
//    //			<string name="filename" value="smoke.vol"/>
//    //		</volume>

//    //		<volume name="albedo" type="constvolume">
//    //			<spectrum name="value" value="0.9"/>
//    //		</volume>
//    //		<float name="scale" value="100"/>
//    //	</medium>


//    //    <shape type="obj">
//    //		<string name="filename" value="bounds.obj"/>

//    //		<ref name="interior" id="smoke"/>
//    //	</shape>

//}

//QDomElement MitsubaExporter::appendShape(QDomElement node, SceneNode * sceneNode)
//{
//    // appends renderables stored in a sceneNode (but ignores children!)

//QDomElement MitsubaExporter::appendShape(QDomElement node, SceneNode * sceneNode)
//{
    // appends renderables stored in a sceneNode (but ignores children!)

    /*
     *
     * Currently, sceneNodes can have multiple renderables attached. ImplicitColliders
     * are renderables with implicit shape types enumerated. This presents some difficulty when
     * figuring out where to put the obj string attribute - in sceneNode or renderables?
     * as a temporary solution, the obj attribute will be put in a sceneNode and we are enforcing that
     * implicitcolliders are sitting in their own scenenode (the obj attribute is added upon collider creation)
     */
    // append <shape> to scene
//    QDomElement s = appendOBJ( node, sceneNode->getObjFile() );

//    // add transformation to the shape node
//    appendXform(s, sceneNode->getCTM());

//    // add default material to the shape node
//    appendBSDF(s);

//    if (sceneNode->getType() == IMPLICIT_COLLIDER)
//    {
//        // TODO
//    }
//    else if (sceneNode->getType() == SNOW_CONTAINER)
//    {
//        // TODO
//    }
//}

//    /*
//     *
//     * Currently, sceneNodes can have multiple renderables attached. ImplicitColliders
//     * are renderables with implicit shape types enumerated. This presents some difficulty when
//     * figuring out where to put the obj string attribute - in sceneNode or renderables?
//     * as a temporary solution, the obj attribute will be put in a sceneNode and we are enforcing that
//     * implicitcolliders are sitting in their own scenenode (the obj attribute is added upon collider creation)
//     */
//    // append <shape> to scene
//    QDomElement s = appendOBJ(node, sceneNode->getObjFile());

//    // add transformation to the shape node
//    appendXform(s, sceneNode->getCTM());

//    // add default material to the shape node
//    appendBSDF(s);

//    if (sceneNode->getType() == IMPLICIT_COLLIDER)
//    {
//        // TODO
//    }
//    else if (sceneNode->getType() == SNOW_CONTAINER)
//    {
//        // TODO
//    }
//}

//QDomElement MitsubaExporter::appendOBJ(QDomElement node, QString objfile)
//{
//    QDomElement s = m_document.createElement("shape");
//    s.setAttribute("type", "obj");
//    QDomElement str = m_document.createElement("string");
//    str.setAttribute("name","filename");
//    str.setAttribute("value",objfile);
//    s.appendChild(str);
//}

//QDomElement MitsubaExporter::appendBSDF(QDomElement node)
//{
//    QDomElement bsdf = m_document.createElement("bsdf");
//    bsdf.setAttribute("type","diffuse");
//    QDomElement rgb = m_document.createElement("rgb");
//    rgb.setAttribute("name","diffuseReflectance");
//    rgb.setAttribute("value",".2, .2, .3"); // dull gray color
//    bsdf.appendChild(rgb);
//    node.appendChild(bsdf);
//    return bsdf;
//}

//QDomElement MitsubaExporter::appendRenderer(QDomElement node)
//{
//    //    <integrator type="volpath_simple">
//    //		<integer name="maxDepth" value="8"/>
//    //	</integrator>
//        // integrator settings
//    QDomElement integrator = m_document.createElement("integrator");
//    integrator.setAttribute("type","volpath_simple");
//    QDomElement in = m_document.createElement("integer");
//    in.setAttribute("name", "maxDepth");
//    in.setAttribute("value","8");
//    integrator.appendChild(in);
//    node.appendChild(integrator);
//    return integrator;
//}

//QDomElement MitsubaExporter::appendXform(QDomElement node, glm::mat4 xform)
//{
//    QDomElement t = m_document.createElement("transform");
//    t.setAttribute("name", "toWorld");
//    QDomElement m = m_document.createElement("matrix");
//    // convert matrix

//    char buf[16*sizeof(float)];
//    sprintf(buf, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",xform[0][0],xform[1][0],xform[2][0],xform[3][0],
//                                                                     xform[0][1],xform[1][1],xform[2][1],xform[3][1],
//                                                                     xform[0][2],xform[1][2],xform[2][2],xform[3][2],
//                                                                     xform[0][3],xform[1][3],xform[2][3],xform[3][3]);
//    QString matstr(buf);
//    m.setAttribute("value", matstr);

//    t.appendChild(m);
//    node.appendChild(t);

//    return t;
//}
