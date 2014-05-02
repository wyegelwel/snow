/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   engine.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#include <GL/gl.h>

#include "common/common.h"
#include "io/mitsubaexporter.h"
#include "sim/caches.h"
#include "sim/implicitcollider.h"
#include "sim/engine.h"
#include "sim/particlesystem.h"
#include "sim/particlegrid.h"
#include "sim/particlegridnode.h"
#include "ui/uisettings.h"

#include "cuda/functions.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define TICKS 10

Engine::Engine()
    : m_particleSystem(NULL),
      m_particleGrid(NULL),
      m_time(0.f),
      m_running(false),
      m_paused(false),
      m_exporter(NULL)
{
    m_particleSystem = new ParticleSystem;
    m_particleGrid =  new ParticleGrid;

    assert( connect(&m_ticker, SIGNAL(timeout()), this, SLOT(update())) );
}

Engine::~Engine()
{
    if ( m_running ) stop();
    SAFE_DELETE( m_particleSystem );
    SAFE_DELETE( m_particleGrid );
    SAFE_DELETE( m_exporter );
}

void Engine::setGrid(const Grid &grid)
{
    m_grid = grid;
    m_particleGrid->setGrid( grid );
}

void Engine::addCollider(const ColliderType &t, const vec3 &center, const vec3 &param, const vec3 &velocity) {
    const ImplicitCollider &col = ImplicitCollider(t,center,param,velocity);
    m_colliders += col;
}

void Engine::addParticleSystem( const ParticleSystem &particles )
{
    QVector<Particle> parts = particles.getParticles();
//    for(int i = 0; i < parts.size(); i++)  {
//        std::cout << "velocity in engine: " << mag << std::endl;
//        parts[i].velocity = vel*mag;
//    }
    *m_particleSystem += particles;
}

void Engine::clearParticleSystem()
{
    m_particleSystem->clear();
}

void Engine::clearParticleGrid()
{
    m_particleGrid->clear();
}

void Engine::initExporter( QString fprefix )
{
    m_exporter = new MitsubaExporter( fprefix, UiSettings::exportFPS() );
}

bool Engine::start( bool exportVolume )
{
    if ( m_particleSystem->size() > 0 && !m_grid.empty() && !m_running ) {
        if ( (m_export = exportVolume) ) m_exporter->reset( m_grid );

        initializeCudaResources();
        m_running = true;

        LOG( "SIMULATION STARTED" );

        m_ticker.start(TICKS);
        return true;

    } else {

        if ( m_particleSystem->size() == 0 ) {
            LOG( "Empty particle system." );
        }
        if ( m_grid.empty() ) {
            LOG( "Empty simulation grid." );
        }
        if ( m_running ) {
            LOG( "Simulation already running." );
        }
        return false;

    }
}

void Engine::stop()
{
    LOG( "SIMULATION STOPPED" );
    m_ticker.stop();
    freeCudaResources();
    m_running = false;
}

void Engine::pause()
{
    m_ticker.stop();
    m_paused = true;
}

void Engine::resume()
{
    if ( m_paused ) {
        m_paused = false;
        if ( m_running ) m_ticker.start(TICKS);
    }
}

void Engine::reset()
{
    if ( !m_running ) {
        clearColliders();
        clearParticleSystem();
        clearParticleGrid();
        m_time = 0.f;
    }
}

bool Engine::isRunning()
{
    return m_running;
}

void Engine::update()
{
    if ( !m_busy && m_running && !m_paused ) {

        m_busy = true;

        cudaGraphicsMapResources( 1, &m_particlesResource, 0 );
        Particle *devParticles;
        size_t size;
        checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**)&devParticles, &size, m_particlesResource ) );
        checkCudaErrors( cudaDeviceSynchronize() );

        if ( (int)(size/sizeof(Particle)) != m_particleSystem->size() ) {
            LOG( "Particle resource error : %lu bytes (%lu expected)", size, m_particleSystem->size()*sizeof(Particle) );
        }

        cudaGraphicsMapResources( 1, &m_nodesResource, 0 );
        Node *devNodes;
        checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**)&devNodes, &size, m_nodesResource ) );
        checkCudaErrors( cudaDeviceSynchronize() );

        if ( (int)(size/sizeof(Node)) != m_particleGrid->size() ) {
            LOG( "Grid nodes resource error : %lu bytes (%lu expected)", size, m_particleGrid->size()*sizeof(Node) );
        }

        updateParticles( devParticles, m_devParticleCaches, m_particleSystem->size(), m_devGrid,
                         devNodes, m_devNodeCaches, m_grid.nodeCount(), m_devColliders, m_colliders.size(),
                         UiSettings::timeStep(), UiSettings::implicit() );

//        updateColliders(); //updating collider positions on cpu side

        if (m_export && (m_time - m_exporter->getLastUpdateTime() >= m_exporter->getspf()))
        {
            cudaMemcpy(m_exporter->getNodesPtr(), devNodes, m_grid.nodeCount() * sizeof(Node), cudaMemcpyDeviceToHost);
            m_exporter->runExportThread(m_time);
        }

        checkCudaErrors( cudaGraphicsUnmapResources( 1, &m_particlesResource, 0 ) );
        checkCudaErrors( cudaGraphicsUnmapResources( 1, &m_nodesResource, 0 ) );
        checkCudaErrors( cudaDeviceSynchronize() );

        m_time += UiSettings::timeStep();

        if (m_time >= UiSettings::maxTime()) // user can adjust max export time dynamically
        {
            stop();
            LOG( "Simulation Completed" );
        }

        m_busy = false;

    } else {

        if ( !m_running ) {
            LOG( "Simulation not running..." );
        }
        if ( m_paused ) {
            LOG( "Simulation paused..." );
        }

    }
}

//void Engine::updateColliders()  {
//    float timestep = UiSettings::timeStep();
//    for(int i = 0; i < m_colliders.size(); i++)  {
//        ImplicitCollider &col = m_colliders[i];
//        col.center += col.velocity*timestep;
//        std::cout << "col vel: " << col.velocity.y << std::endl;
//    }
//}

void Engine::initializeCudaResources()
{
    LOG( "Initializing CUDA resources..." );

    // Particles
    registerVBO( &m_particlesResource, m_particleSystem->vbo() );
    float particlesSize = m_particleSystem->size()*sizeof(Particle) / 1e6;
    LOG( "Allocated %.2f MB for particle system.", particlesSize );

    int numNodes = m_grid.nodeCount();

    // Grid Nodes
    registerVBO( &m_nodesResource, m_particleGrid->vbo() );
    float nodesSize =  numNodes*sizeof(Node) / 1e6;
    LOG( "Allocating %.2f MB for grid nodes.", nodesSize );

    // Grid
    checkCudaErrors(cudaMalloc( (void**)&m_devGrid, sizeof(Grid) ));
    checkCudaErrors(cudaMemcpy( m_devGrid, &m_grid, sizeof(Grid), cudaMemcpyHostToDevice ));


    // Colliders
    checkCudaErrors(cudaMalloc( (void**)&m_devColliders, m_colliders.size()*sizeof(ImplicitCollider) ));
    checkCudaErrors(cudaMemcpy( m_devColliders, m_colliders.data(), m_colliders.size()*sizeof(ImplicitCollider), cudaMemcpyHostToDevice ));

    // Caches
    checkCudaErrors(cudaMalloc( (void**)&m_devNodeCaches, numNodes*sizeof(NodeCache)) );
    checkCudaErrors(cudaMemset( m_devNodeCaches, 0, numNodes*sizeof(NodeCache)) );
    float nodeCachesSize = numNodes*sizeof(NodeCache) / 1e6;
    LOG( "Allocating %.2f MB for implicit update node cache.", nodeCachesSize );

    checkCudaErrors(cudaMalloc( (void**)&m_devParticleCaches, m_particleSystem->size()*sizeof(ParticleCache) ));
    checkCudaErrors(cudaMemset( m_devParticleCaches, 0, m_particleSystem->size()*sizeof(ParticleCache)));
    float particleCachesSize = m_particleSystem->size()*sizeof(ParticleCache) / 1e6;
    LOG( "Allocating %.2f MB for implicit update particle caches.", particleCachesSize );

    LOG( "Allocated %.2f MB in total", particlesSize + nodesSize + nodeCachesSize + particleCachesSize );

    LOG( "Computing particle volumes..." );
    cudaGraphicsMapResources( 1, &m_particlesResource, 0 );
    Particle *devParticles;
    size_t size;
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**)&devParticles, &size, m_particlesResource ) );
    if ( (int)(size/sizeof(Particle)) != m_particleSystem->size() ) {
        LOG( "Particle resource error : %lu bytes (%lu expected)", size, m_particleSystem->size()*sizeof(Particle) );
    }
    initializeParticleVolumes( devParticles, m_particleSystem->size(), m_devGrid, numNodes );
    checkCudaErrors( cudaGraphicsUnmapResources(1, &m_particlesResource, 0) );

    LOG( "Initialization complete." );
}

void Engine::freeCudaResources()
{
    LOG( "Freeing CUDA resources..." );
    unregisterVBO( m_particlesResource );
    unregisterVBO( m_nodesResource );
    cudaFree( m_devGrid );
    cudaFree( m_devColliders );
    cudaFree( m_devNodeCaches );
    cudaFree( m_devParticleCaches );
    cudaFree( m_devMaterial );
}

void Engine::render()
{
    if ( UiSettings::showParticles() ) m_particleSystem->render();
    if ( UiSettings::showGridData() && m_running ) m_particleGrid->render();
}

BBox Engine::getBBox( const glm::mat4 &ctm )
{
    return m_particleGrid->getBBox( ctm );
}

vec3 Engine::getCentroid( const glm::mat4 &ctm )
{
    return m_particleGrid->getCentroid( ctm );
}
