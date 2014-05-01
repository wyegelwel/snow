/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scene.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef SCENE_H
#define SCENE_H

#include "glm/mat4x4.hpp"

class ParticleSystem;
class Renderable;
class SceneNode;
class SceneNodeIterator;
class QString;

class Scene
{

public:

    Scene();
    virtual ~Scene();

    virtual void render();

    virtual void renderVelocity(bool velTool);

    SceneNode* root() { return m_root; }

    SceneNode* getSceneGridNode();

    SceneNodeIterator begin() const;

    void deleteSelectedNodes();

    void loadMesh(const QString &filename, glm::mat4 CTM=glm::mat4());

    void reset();
    void initSceneGrid();
    void updateSceneGrid();

private:

    SceneNode *m_root;

    void setupLights();

};

#endif // SCENE_H
