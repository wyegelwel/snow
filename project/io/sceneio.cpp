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

SceneIO::SceneIO()
{

}

bool SceneIO::read(QString fname, Scene *scene, Engine *engine)
{
    m_scenefile = fname;

}

bool SceneIO::write(QString fname, Scene *scene, Engine *engine)
{

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
}
