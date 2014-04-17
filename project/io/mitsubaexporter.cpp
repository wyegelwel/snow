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

MitsubaExporter::MitsubaExporter()
{
}

void MitsubaExporter::exportVolumeData(QString fprefix, BBox bounds)
{
    // TODO - need to export the grid.

    // add .vol as extension


    // std::string path = dir +
        //std::ofstream o();
       // Float scale = 1.0f / std::max(std::max(xres, yres), zres);

    //        os.write("VOL", 3);
    //        char version = 3;
    //        os.write((char *) &version, sizeof(char));
    //        int value = 1;
    //        os.write((char *) &value, sizeof(int));
    //        os.write((char *) &xres, sizeof(int));
    //        os.write((char *) &yres, sizeof(int));
    //        os.write((char *) &zres, sizeof(int));
    //        value = 1;
    //        os.write((char *) &value, sizeof(int));

    //        float minX = -xres/2.0f*scale;
    //        float minY = -yres/2.0f*scale;
    //        float minZ = -zres/2.0f*scale;
    //        float maxX = xres/2.0f*scale;
    //        float maxY = yres/2.0f*scale;
    //        float maxZ = zres/2.0f*scale;

    //        os.write((char *) &minX, sizeof(float));
    //        os.write((char *) &minY, sizeof(float));
    //        os.write((char *) &minZ, sizeof(float));
    //        os.write((char *) &maxX, sizeof(float));
    //        os.write((char *) &maxY, sizeof(float));
    //        os.write((char *) &maxZ, sizeof(float));
    //        for (size_t i=0; i<density.size(); ++i) {
    //            float value = density[i].density; // this is a single float representing the density
    //            os.write((char *) &value, sizeof(float));
    //        }
    //        os.close();
}

void MitsubaExporter::exportScene(QString fprefix, int frame, Scene *scene)
{
    /**
     * Exports Single Frame of Scene to Mitsuba format.
     *
     * Traverses the scene graph and converts colliders to basic Mitsuba primitives
     * filled objects are loaded OBJs, and the snow is a heterogenous medium-type volume data
     *
     */
    //QString fname = fprefix + QString("_") + QString::number(frame,);
    QString fname = QString("%1_%2.xml").arg(fprefix, QString("%1").arg(frame,4,'d',0,'0'));
    QDomDocument document;

    // xml header
    QDomProcessingInstruction pi = document.createProcessingInstruction("xml", "version=\"1.0\" encoding=\"utf-8\" ");
    document.appendChild(pi);

    // root element
    QDomElement sceneNode = document.createElement("scene");
    sceneNode.setAttribute("version", "0.5.0");
    document.appendChild(sceneNode);

    //addRenderer(document, sceneNode);
    //addCamera(document, sceneNode);

    // now traverse the scene graph for renderables.
//    for (int i=0; i<scene->root())



    /// NOTE: it appears that shapes are not nested in the Mitsuba format.
    /// so we have to compute CTM for each shape node?

    // Adding more elements
//    for(int i = 0; i < 5; i++)
//    {
//        QDomElement dorm = document.createElement("Dorm");
//        dorm.setAttribute("Name", "Dorm Building " + QString::number(i));
//        dorm.setAttribute("ID", QString::number(i));
//        root.appendChild(dorm);
//    }


    // write it to file
    QFile file(fname);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        std::cout << "write failed" << std::endl;
    }
    else
    {
        QTextStream stream(&file);
        stream << document.toString();
        file.close();
        std::cout << "file written" << std::endl;
    }

}

void MitsubaExporter::addCollider(QDomDocument &doc, QDomElement &node)
{
//    <shape type="obj">
//		<!-- Shiny floor -->
//		<string name="filename" value="plane.obj"/>

//		<bsdf type="diffuse">
//			<rgb name="diffuseReflectance" value=".2, .2, .3"/>
//		</bsdf>
//		<transform name="toWorld">
//			<translate y=".48"/>
//		</transform>
//	</shape>


//    <shape type="sphere">
//		<point name="center" x="0" y="-2" z="-1"/>
//		<float name="radius" value=".2"/>

//		<emitter type="area">
//			<spectrum name="radiance" value="400"/>
//		</emitter>
//	</shape>

}

void MitsubaExporter::addRenderer(QDomDocument &doc, QDomElement &node)
{
//    <integrator type="volpath_simple">
//		<integer name="maxDepth" value="8"/>
//	</integrator>
    // integrator settings
//    QDomElement integratorNode = document.createElement("integrator");
//    integratorNode.setAttribute("maxDepth", "8");
//    node.appendChild(integratorNode);
}


void MitsubaExporter::addMedium(QDomDocument &doc, QDomElement &node)
{

//    <medium type="heterogeneous" id="smoke">
//		<string name="method" value="woodcock"/>

//		<volume name="density" type="gridvolume">
//			<string name="filename" value="smoke.vol"/>
//		</volume>

//		<volume name="albedo" type="constvolume">
//			<spectrum name="value" value="0.9"/>
//		</volume>
//		<float name="scale" value="100"/>
//	</medium>


//    <shape type="obj">
//		<string name="filename" value="bounds.obj"/>

//		<ref name="interior" id="smoke"/>
//	</shape>

}

void MitsubaExporter::addCamera(QDomDocument &doc, QDomElement &node)
{
//    <sensor type="perspective">
//		<float name="focusDistance" value="1.25668"/>
//		<float name="fov" value="45.8402"/>
//		<string name="fovAxis" value="x"/>
//		<transform name="toWorld">
//			<scale x="-1"/>

//			<lookat target="-0.166029, 0.148984, -0.537402" origin="-0.61423, 0.154197, -1.43132" up="-0.000640925, -0.999985, -0.0055102"/>
//		</transform>

//		<sampler type="ldsampler">
//			<integer name="sampleCount" value="64"/>
//		</sampler>

//		<film type="hdrfilm">
//			<integer name="height" value="576"/>
//			<integer name="width" value="768"/>

//			<rfilter type="gaussian"/>
//		</film>
//	</sensor>


//    QDomElement sensorNode = document.createElement("sensor");
//    sensorNode.setAttribute("type","perspective"); {
//        QDomElement fd = document.createElement("float");
//        fd.setAttribute("focusDistance", "1.25668");
//        sensorNode.appendChild(fd);

//        QDomElement fov = document.createElement("float");
//        fov.setAttribute("fov", "45.8402");
//        sensorNode.appendChild(fov);

//        QDomElement fovAxis = document.createElement("string");
//        fovAxis.setAttribute("fovAxis","x");
//        sensorNode.appendChild(fovAxis);

        // transform node
        // sampler node
        // film node

//    }
//    sceneNode.appendChild(sensorNode);

}
