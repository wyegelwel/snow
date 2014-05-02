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
#include "geometry/mesh.h"
#include "ui/uisettings.h"
#include "cuda/vector.h"
#include "scene/scenecollider.h"
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
    m_document.clear();

    QFileInfo info(filename);
    m_sceneFilePrefix = QString("%1/%2").arg(info.absolutePath(),info.baseName());

    QFile* file = new QFile(filename);
    if (!file->open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox msgBox;
        msgBox.setText("Error : Invalid XML file");
        msgBox.exec();
        return false;
    }

    engine->reset();
    scene->reset();
    scene->initSceneGrid();

    QString errMsg; int errLine; int errCol;
    if (!m_document.setContent(file, &errMsg, &errLine, &errCol))
    {
        QMessageBox msgBox;
        errMsg = QString("XML Import Error : Line %1, Col %2 : %3").arg(QString::number(errLine),QString::number(errCol),errMsg);
        msgBox.setText(errMsg);
        msgBox.exec();
        return false;
    }

    applySimulationParameters();
    applyExportSettings();
    applyParticleSystem(scene);
    applyGrid(scene);
    applyColliders(scene, engine);
}

void SceneIO::applySimulationParameters()
{
    QDomNodeList nlist = m_document.elementsByTagName("SimulationParameters");
    QDomElement sNode = nlist.at(0).toElement();
    QDomNodeList sList = sNode.childNodes();
    for (int i=0; i<sList.size(); ++i)
    {
        QDomElement n = sList.at(i).toElement();

        if (n.attribute("name").compare("timeStep") == 0)
        {
            bool ok;
            float ts = n.attribute("value").toFloat(&ok);
            if (ok)
                UiSettings::timeStep() = ts;
        }
    }
}

void SceneIO::applyExportSettings()
{
    QDomNodeList list = m_document.elementsByTagName("ExportSettings");
    QDomElement s = list.at(0).toElement();
    for (int i=0; i<s.childNodes().size(); ++i)
    {
        QDomElement e = s.childNodes().at(i).toElement();
        QString name = e.attribute("name");

        if (name.compare("filePrefix") == 0)
            m_sceneFilePrefix = e.attribute("value");
        else if (name.compare("maxTime") == 0)
            UiSettings::maxTime() = e.attribute("value").toInt();
        else if (name.compare("exportFPS") == 0)
            UiSettings::exportFPS() = e.attribute("value").toInt();
        else if (name.compare("exportDensity") == 0)
            UiSettings::exportVelocity() = e.attribute("value").toInt();
        else if (name.compare( "exportVelocity") == 0)
            UiSettings::exportVelocity() = e.attribute("value").toInt();
    }
}

void SceneIO::applyParticleSystem(Scene *scene)
{
    // does not call fillParticles for the user.
    QDomNodeList list = m_document.elementsByTagName("SnowContainer");
    int numParticles;
    QString fname;
    int materialPreset;
    glm::mat4 CTM;
    for (int s=0; s<list.size(); ++s)
    {
        // for each SnowContainer, import the obj into the scene
        QDomElement p = list.at(s).toElement();
        for (int t=0; t<p.childNodes().size(); ++t)
        {
            QDomElement d = p.childNodes().at(t).toElement();
            QString name = d.attribute("name");
            if (name.compare("numParticles") == 0)
                numParticles = d.attribute("value").toInt(); // we are not doign anythign with these right now
            else if (name.compare("filename") == 0)
                fname = d.attribute("value");
            else if (name.compare("materialPreset") == 0)
                 materialPreset = d.attribute("value").toInt();
            else if (name.compare("CTM") == 0)
            {
                QStringList floatWords = d.attribute("value").split(QRegExp("\\s+"));
                int k=0;
                for (int i=0; i<4; i++)
                    for (int j=0; j<4; j++,k++)
                        CTM[j][i] = floatWords.at(k).toFloat();
            }

        }
        scene->loadMesh(fname, CTM);
    }
}

void SceneIO::applyGrid(Scene * scene)
{
    Grid grid;
    QDomNodeList list = m_document.elementsByTagName("Grid");
    QDomElement g = list.at(0).toElement();
    for (int i=0; i < g.childNodes().size(); ++i)
    {
        QDomElement e = g.childNodes().at(i).toElement();
        QString name = e.attribute("name");
        if (name.compare("gridDim") == 0 )
        {
            UiSettings::gridDimensions() = glm::ivec3(e.attribute("x").toInt(),e.attribute("y").toInt(), e.attribute("z").toInt());
        }
        else if (name.compare("pos") == 0 )
        {
            UiSettings::gridPosition() = vec3(e.attribute("x").toFloat(), e.attribute("y").toFloat(), e.attribute("z").toFloat());
        }
        else if (name.compare("h") == 0)
        {
            UiSettings::gridResolution() = e.attribute("value").toFloat();
        }
    }
    scene->updateSceneGrid();
}

void SceneIO::applyColliders(Scene * scene, Engine * engine)
{
    vec3 center, velocity, param;
    int colliderType;

    QDomNodeList list = m_document.elementsByTagName("Collider");
    for (int i=0; i<list.size(); ++i)
    {
        QDomElement e = list.at(i).toElement();
        colliderType = e.attribute("type").toInt();
        for (int j=0; j<e.childNodes().size(); j++)
        {
            QDomElement c = e.childNodes().at(j).toElement();
            vec3 vector;
            vector.x = c.attribute("x").toFloat();
            vector.y = c.attribute("y").toFloat();
            vector.z = c.attribute("z").toFloat();
            QString name = c.attribute("name");
            if (name.compare("center")==0)
            {
                center = vector;
            }
            else if (name.compare("velocity")==0)
            {
                velocity = vector;
            }
            else if (name.compare("param")==0)
            {
                param = vector;
            }
        }
        scene->addCollider((ColliderType)colliderType, center, param, velocity);
        engine->addCollider((ColliderType)colliderType, center, param, velocity);
    }
}



bool SceneIO::write(Scene *scene, Engine *engine)
{
    m_document.clear();
    QDomProcessingInstruction processInstruct = m_document.createProcessingInstruction("xml", "version=\"1.0\" encoding=\"utf-8\" ");
    m_document.appendChild(processInstruct);

    QDomElement root = m_document.createElement("SnowSimulation"); // root node of the scene
    m_document.appendChild(root);

    appendSimulationParameters(root, UiSettings::timeStep());
    appendExportSettings(root);
    appendParticleSystem(root, scene);
    appendGrid(root, scene);
    appendColliders(root, scene);

    QString fname = QString("%1.xml").arg(m_sceneFilePrefix);
    QFile file(fname);
    if (!file.open(QIODevice::ReadWrite | QIODevice::Truncate | QIODevice::Text))
    {
        LOG("write failed!");
    }
    else
    {
        QTextStream stream(&file);
        int indent = 4;
        stream << m_document.toString(indent);
        file.close();
        LOG("file written!");
    }
}

void SceneIO::appendColliders(QDomElement root, Scene * scene)
{
    int count = 0;
    QDomElement icNode = m_document.createElement("ImplicitColliders");

    vec3 velocity;
    for ( SceneNodeIterator it = scene->begin(); it.isValid(); ++it )
    {
        if ((*it)->hasRenderable() && (*it)->getType() == SceneNode::SCENE_COLLIDER)
        {
            QDomElement cNode = m_document.createElement("Collider");
            SceneCollider * sCollider = dynamic_cast<SceneCollider*>((*it)->getRenderable());
            ImplicitCollider iCollider(*sCollider->getImplicitCollider()); // make copy

            iCollider.applyTransformation((*it)->getCTM());
            if(!EQ(sCollider->getVelMag(),0)) {
//                glm::vec4 vel = (*it)->getCTM()*glm::vec4(sCollider->getVelVec(),1.f);
//                iCollider.velocity = vec3::normalize(vec3(vel.x,vel.y,vel.z))*sCollider->getVelMag();
                iCollider.velocity = sCollider->getVelMag() * sCollider->getWorldVelVec((*it)->getCTM());
            }
            else iCollider.velocity = vec3(0,0,0);

            cNode.setAttribute("type", iCollider.type);
            appendVector(cNode, "center", iCollider.center);
            appendVector(cNode, "velocity", iCollider.velocity);
            appendVector(cNode, "param", iCollider.param);
            icNode.appendChild(cNode);

            count++;
        }
    }
    if (count > 0)
        root.appendChild(icNode);
}

void SceneIO::appendExportSettings(QDomElement root)
{
    QDomElement eNode = m_document.createElement("ExportSettings");
    appendString(eNode, "filePrefix", m_sceneFilePrefix);
    appendInt(eNode, "maxTime", UiSettings::maxTime() );
    appendInt(eNode, "exportFPS", UiSettings::exportFPS() );
    appendInt(eNode, "exportDensity", UiSettings::exportDensity());
    appendInt(eNode, "exportVelocity", UiSettings::exportVelocity());
    root.appendChild(eNode);
}

void SceneIO::appendGrid(QDomElement root, Scene * scene)
{
    // ENGINE grid does not reflect grid until start() button is pressed.
    // find the SIMULATION_GRID sceneNode object
    Grid grid;
    SceneNode * gridNode = scene->getSceneGridNode();
    grid = UiSettings::buildGrid(gridNode->getCTM());
    QDomElement gNode = m_document.createElement("Grid");
    appendDim(gNode, "gridDim", grid.dim);
    appendVector(gNode, "pos", grid.pos);
    appendFloat(gNode, "h", grid.h);
    root.appendChild(gNode);
}

void SceneIO::appendParticleSystem(QDomElement root, Scene * scene)
{
    int count = 0;
    QDomElement pNode = m_document.createElement("ParticleSystem");
    for ( SceneNodeIterator it = scene->begin(); it.isValid(); ++it )
    {
        if ((*it)->hasRenderable() && (*it)->getType() == SceneNode::SNOW_CONTAINER)
        {
            QDomElement cNode = m_document.createElement("SnowContainer");
            Mesh * mesh = dynamic_cast<Mesh*>((*it)->getRenderable());
            appendString(cNode,"filename",mesh->getFilename());
            appendMatrix(cNode, "CTM", (*it)->getCTM());
            pNode.appendChild(cNode);
            count++;
        }
    }
    if (count == 0)
        return;
    root.appendChild(pNode);
}

void SceneIO::appendSimulationParameters(QDomElement root, float timeStep)
{
    QDomElement spNode = m_document.createElement("SimulationParameters");
    appendFloat(spNode, "timeStep", timeStep);
    root.appendChild(spNode);
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

