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
#include "sim/parameters.h"

struct cudaGraphicsResource;
class ParticleGridTempData;

class Engine : public QObject
{

    Q_OBJECT

public:

    Engine();
    virtual ~Engine();

    void start();
    void pause();
    void resume();
    void stop();

    float getSimulationTime() { return m_time; }

    SimulationParameters& parameters() { return m_params; }

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

    SimulationParameters m_params;
    float m_time;

    bool m_busy;
    bool m_running;
    bool m_paused;

    void initializeCudaResources();
    void freeCudaResources();
};

#endif // ENGINE_H
