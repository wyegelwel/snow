/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scene.h
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef SCENE_H
#define SCENE_H

class SceneNode;


class Scene
{

public:

    Scene();
    virtual ~Scene();

    virtual void render();

    SceneNode* root() { return m_root; }


private:
    //WorldParams m_params; // parameters of the simulation.
    SceneNode *m_root;

};

#endif // SCENE_H
