/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   engine.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#ifndef ENGINE_H
#define ENGINE_H

/**
 * @brief class that handles simulation and update of the ParticleSystem.
 * Simulates the ParticleSystem without drawing
 */

#include <QFile>

class Scene;
class ParticleSystem;

class Engine
{
public:
    Engine();

    /**
     * loads simulation from XML file
     * calls the SceneParser class to do this
     */
    void load(QFile file);

    /**
     * Runs the simulation
     */
    void start();

    /**
     * calls MitsubaExporter class to serialize volume data
     * also writes out the collider primitives to Mitsuba-compatible shapes
     */
    void exportMitsuba();

    /**
     * called to advance particles 1 time step.
     */
    void update();

    int numParticles();

private:
    ParticleSystem *m_particles;
    Scene *m_scene;
    bool m_paused; // pause particle update
};

#endif // ENGINE_H
