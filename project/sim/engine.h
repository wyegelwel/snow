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

#include "common/renderable.h"
#include "geometry/grid.h"
#include "sim/collider.h"
#include "sim/material.h"
#include "sim/parameters.h"
#include "geometry/grid.h"
#include "io/mitsubaexporter.h"

struct cudaGraphicsResource;
struct Particle;
struct ParticleGrid;
struct Node;

struct NodeCache;
struct ParticleCache;

class Engine : public QObject, public Renderable
{

    Q_OBJECT

public:

    Engine();
    virtual ~Engine();

    // Returns whether it actually did start
    bool start( bool exportScene );
    void pause();
    void resume();
    void stop();
    void reset();

    float getSimulationTime() { return m_time; }

    SimulationParameters& parameters() { return m_params; }

    void addParticleSystem( const ParticleSystem &particles );
    void clearParticleSystem();
    ParticleSystem* particleSystem() { return m_particleSystem; }

    void setGrid( const Grid &grid );
    void clearParticleGrid();

    MaterialConstants& materialConstants() { return m_materialConstants; }

//    void addCollider( const ImplicitCollider &collider ) { m_colliders += collider; }
    void addCollider(Collider &collider, const glm::mat4 &ctm);
    void clearColliders() { m_colliders.clear(); }
    QVector<ImplicitCollider>& colliders() { return m_colliders; }

    void initExporter(QString fprefix);

    bool isRunning();

    virtual void render();

    virtual BBox getBBox( const glm::mat4 &ctm );
    virtual vec3 getCentroid( const glm::mat4 &ctm );

public slots:

    void update();

private:

    QTimer m_ticker;

    // CPU data structures
    ParticleSystem *m_particleSystem;
    ParticleGrid *m_particleGrid;
    Grid m_grid;
    QVector<ImplicitCollider> m_colliders;
    MaterialConstants m_materialConstants;

    // CUDA pointers
    cudaGraphicsResource *m_particlesResource; // Particles
    cudaGraphicsResource *m_nodesResource; // Particle grid nodes
    Grid *m_devGrid;

    NodeCache *m_devNodeCache;
    ParticleCache *m_devParticleCaches;

    ImplicitCollider *m_devColliders;
    MaterialConstants *m_devMaterial;

    SimulationParameters m_params;
    float m_time;

    bool m_busy;
    bool m_running;
    bool m_paused;
    bool m_export;

    MitsubaExporter * m_exporter;

    void initializeCudaResources();
    void freeCudaResources();

};

#endif // ENGINE_H
