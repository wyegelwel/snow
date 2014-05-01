/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   sceneio.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#include "sceneio.h"

#include <QFileDialog>
#include <QMessageBox>

#include "sim/engine.h"
#include "sim/particlesystem.h"
#include "scene/scene.h"
#include "scene/scenenodeiterator.h"

#include "ui/uisettings.h"
#include "cuda/vector.h"

#include "glm/gtx/string_cast.hpp"

#include "common/common.h"

SceneIO::SceneIO()
{

}

void SceneIO::setSceneFile(QString filename)
{
    // turn filename into an absolute path
    QFileInfo info(filename);
    m_sceneFilePrefix = QString("%1/%2").arg(info.absolutePath(),info.baseName());
}

bool SceneIO::read(QString filename, Scene *scene, Engine *engine)
{
    QFileInfo info(filename);
    m_sceneFilePrefix = QString("%1/%2").arg(info.absolutePath(),info.baseName());
    // this one is a bit trickier, maybe need access to private members of Engine and Scene?
}

bool SceneIO::write(Scene *scene, Engine *engine)
{
    QDomProcessingInstruction processInstruct = m_document.createProcessingInstruction("xml", "version=\"1.0\" encoding=\"utf-8\" ");
    m_document.appendChild(processInstruct);

    appendSimulationParameters(engine->parameters());
    appendExportSettings();
    appendParticleSystem(scene);
    appendGrid(engine->getGrid());
    appendColliders(engine->colliders());
    // get timeStep from engine->m_params->timestep
    //

    //    // xml header
    //    QDomProcessingInstruction pi = m_document.createProcessingInstruction("xml", "version=\"1.0\" encoding=\"utf-8\" ");
    //    m_document.appendChild(pi);
    //    // root element for the scene
    //    QDomElement sceneNode = m_document.createElement("scene");
    //    sceneNode.setAttribut( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it )e("version", "0.5.0");
    //    m_document.appendChild(sceneNode);

    //    // we want a volumetric path tracer
    //    appendRenderer(sceneNode);
    //    // add the camera
    //    appendCamera(sceneNode, camera);

    QString fname = QString("%1.xml").arg(m_sceneFilePrefix);
    QFile file(fname);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        LOG("write failed!");
    }
    else
    {
        QTextStream stream(&file);
        stream << m_document.toString();
        file.close();
        LOG("file written!");
    }
}

void SceneIO::appendColliders(QVector<ImplicitCollider> colliders)
{
    if (colliders.size() < 1)
        return;
    QDomElement icNode = m_document.createElement("ImplicitColliders");
    for (int i=0; i<colliders.size(); ++i)
    {
        ImplicitCollider collider = colliders[i];
        QDomElement cNode = m_document.createElement("Collider");
        appendVector(cNode, "center", collider.center);
        appendVector(cNode, "velocity", collider.velocity);
        appendVector(cNode, "param", collider.param);
        icNode.appendChild(cNode);
    }

    m_document.appendChild(icNode);
}

void SceneIO::appendExportSettings()
{
    QDomElement eNode = m_document.createElement("ExportSettings");
    appendString(eNode, "filePrefix", m_sceneFilePrefix);
    appendInt(eNode, "numFrames", UiSettings::maxTime() * UiSettings::exportFPS() );
    appendInt(eNode, "exportDensity", UiSettings::exportDensity());
    appendInt(eNode, "exportVelocity", UiSettings::exportVelocity());
    m_document.appendChild(eNode);
}

void SceneIO::appendGrid(Grid grid)
{
    QDomElement gNode = m_document.createElement("Grid");
    appendDim(gNode, "gridDim", grid.dim);
    appendVector(gNode, "pos", grid.pos);
    appendFloat(gNode, "h", grid.h);
    m_document.appendChild(gNode);
}

void SceneIO::appendParticleSystem(Scene * scene)
{
    int count = 0;
    QDomElement pNode = m_document.createElement("ParticleSystem");
    for ( SceneNodeIterator it = scene->begin(); it.isValid(); ++it )
    {
        if ((*it)->hasRenderable() && (*it)->getType() == SceneNode::SNOW_CONTAINER)
        {
            QDomElement cNode = m_document.createElement("Container");
            Mesh * mesh = dynamic_cast<Mesh*>((*it)->getRenderable());
            int matPreset = mesh->getMaterialPreset();
            int numParticles = mesh->getParticleCount();
            appendString(cNode, "materialPreset", QString::number(matPreset));
            appendInt(cNode, "numParticles", numParticles);
            appendMatrix(cNode, "CTM", (*it)->getCTM());
            pNode.appendChild(cNode);
            count++;
        }
    }
    if (count == 0)
        return;
    m_document.appendChild(pNode);
}


void SceneIO::appendSimulationParameters(SimulationParameters params)
{
    QDomElement spNode = m_document.createElement("SimulationParameters");
    appendInt(spNode, "timestep", params.timeStep);
    m_document.appendChild(spNode);
}

void SceneIO::appendDim(QDomElement node, const QString name, const glm::ivec3 iv)
{
    QDomElement dNode = m_document.createElement("dim");
    dNode.setAttribute("name",name);
    dNode.setAttribute("x", iv.x);
    dNode.setAttribute("y", iv.y);
    dNode.setAttribute("z", iv.z);
    node.appendChild(dNode);
}


void SceneIO::appendFloat(QDomElement node, const QString name, const float f)
{
    QDomElement fNode = m_document.createElement("float");
    fNode.setAttribute("name", name);
    fNode.setAttribute("value", f);
    node.appendChild(fNode);
}


void SceneIO::appendInt(QDomElement node, const QString name, const int i)
{
    QDomElement iNode = m_document.createElement("int");
    iNode.setAttribute("name",name);
    iNode.setAttribute("value",i);
    node.appendChild(iNode);
}

void SceneIO::appendMatrix(QDomElement node, const QString name, const glm::mat4 m)
{
    QDomElement mNode = m_document.createElement("matrix");
    mNode.setAttribute("name",name);
    QString matstr;
    matstr.sprintf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
                    m[0][0], m[1][0], m[2][0], m[3][0],
                    m[0][1], m[1][1], m[2][1], m[3][1],
                    m[0][2], m[1][2], m[2][2], m[3][2],
                    m[0][3], m[1][3], m[2][3], m[3][3]);
    mNode.setAttribute("value",matstr);
    node.appendChild(mNode);
}

void SceneIO::appendString(QDomElement node, const QString name, const QString value)
{
    QDomElement sNode= m_document.createElement("string");
    sNode.setAttribute("name", name);
    sNode.setAttribute("value", value);
    node.appendChild(sNode);
}

void SceneIO::appendVector(QDomElement node, const QString name, const vec3 v)
{
    QDomElement vNode = m_document.createElement("vector");
    vNode.setAttribute("name",name);
    vNode.setAttribute("x", v.x);
    vNode.setAttribute("y", v.y);
    vNode.setAttribute("z", v.z);
    node.appendChild(vNode);
}

