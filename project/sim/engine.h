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

#include <QObject>
#include <QTimer>
#include <QVector>

#include "sim/collider.h"
#include "sim/material.h"
#include "sim/particle.h"
#include "sim/particlegrid.h"

struct cudaGraphicsResource;
class ParticleGridTempData;

class Engine : public QObject
{

    Q_OBJECT

public:

    struct Parameters
    {
        float timeStep;
        float startTime;
        float endTime;
        Parameters( float dt, float start, float end ) : timeStep(dt), startTime(start), endTime(end) {}
    };

    Engine();
    virtual ~Engine();

    /**
     * Sets the time to zero and starts the simulation.
     */
    void start();

    /**
     * Pauses the simulation.
     */
    void pause();

    /**
     * Resumes the simulation (without restarting the time).
     */
    void resume();

    void stop();

    void setStartTime( float start ) { m_params.startTime = start; }
    void setEndTime( float end ) { m_params.endTime = end; }
    void setTimeStep( float dt ) { m_params.timeStep = dt; }

    void addParticleSystem( const ParticleSystem &particles ) { *m_particleSystem += particles; }
    void clearParticleSystem() { m_particleSystem->clear(); }
    ParticleSystem* particleSystem() { return m_particleSystem; }

    Grid& grid() { return m_grid; }
    MaterialConstants& materialConstants() { return m_materialConstants; }

    void addCollider( const ImplicitCollider &collider ) { m_colliders += collider; }
    void clearColliders() { m_colliders.clear(); }
    QVector<ImplicitCollider>& colliders() { return m_colliders; }

    /**
     * calls MitsubaExporter class to serialize volume data
     * also writes out the collider primitives to Mitsuba-compatible shapes
     */
    void exportMitsuba() {}

public slots:

    void update();

    ParticleSystem* particleSystem();

private:

    QTimer m_ticker;

    // CPU data structures
    ParticleSystem *m_particleSystem;
    Grid m_grid;
    QVector<ImplicitCollider> m_colliders;
    MaterialConstants m_materialConstants;

    // CUDA pointers
    cudaGraphicsResource *m_cudaResource; // Particles
    Grid *m_devGrid;
    ParticleGrid::Node *m_devNodes;
    ParticleGridTempData *m_devPGTD;
    ImplicitCollider *m_devColliders;
    MaterialConstants *m_devMaterial;

    Parameters m_params;
    float m_time;

    bool m_running;
    bool m_paused;

    void initializeCudaResources();
    void freeCudaResources();

};

#endif // ENGINE_H
