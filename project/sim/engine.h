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

#include "common/renderable.h"
#include "geometry/grid.h"
#include "sim/implicitcollider.h"
#include "sim/material.h"

struct cudaGraphicsResource;

struct Node;
struct NodeCache;
struct Particle;
struct ParticleCache;
struct ParticleGrid;
struct ParticleSystem;

struct MitsubaExporter;

class Engine : public QObject, public Renderable
{

    Q_OBJECT

public:

    Engine();
    virtual ~Engine();

    // Returns whether it actually did start
    bool start( bool exportVolume );
    void pause();
    void resume();
    void stop();
    void reset();

    float getSimulationTime() { return m_time; }

    void addParticleSystem( const ParticleSystem &particles );
    void clearParticleSystem();
    ParticleSystem* particleSystem() { return m_particleSystem; }

    void setGrid( const Grid &grid );
    void clearParticleGrid();

    Grid getGrid() {return m_grid; }

    void initParticleMaterials( int preset );

    void addCollider( const ImplicitCollider &collider ) { m_colliders += collider; }
    void addCollider(const ColliderType &t,const vec3 &center, const vec3 &param, const vec3 &velocity);

    void clearColliders() { m_colliders.clear(); }
//    void updateColliders();
    QVector<ImplicitCollider>& colliders() { return m_colliders; }

    void initExporter( QString fprefix );

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

    // CUDA pointers
    cudaGraphicsResource *m_particlesResource; // Particles
    cudaGraphicsResource *m_nodesResource; // Particle grid nodes
    Grid *m_devGrid;

    NodeCache *m_devNodeCaches;
    ParticleCache *m_devParticleCaches;

    ImplicitCollider *m_devColliders;
    Material *m_devMaterial;

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
