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
class ParticleSystem;

class Scene
{

public:

    Scene();
    virtual ~Scene();

    virtual void render();

    SceneNode* root() { return m_root; }

    void setParticleSystem( ParticleSystem *particleSystem ) { m_particleSystem = particleSystem; }


private:

    SceneNode *m_root;

    ParticleSystem *m_particleSystem;

};

#endif // SCENE_H
