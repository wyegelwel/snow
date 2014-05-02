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

    m_hostParticleCache = NULL;

    assert( connect(&m_ticker, SIGNAL(timeout()), this, SLOT(update())) );
}

Engine::~Engine()
{
    if ( m_running ) stop();
    SAFE_DELETE( m_particleSystem );
    SAFE_DELETE( m_particleGrid );
    SAFE_DELETE( m_hostParticleCache );
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

        updateParticles( devParticles, m_devParticleCache, m_hostParticleCache, m_particleSystem->size(), m_devGrid,
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

void Engine::initializeCudaResources()
{
    LOG( "Initializing CUDA resources..." );

    // Particles
    registerVBO( &m_particlesResource, m_particleSystem->vbo() );
    float particlesSize = m_particleSystem->size()*sizeof(Particle) / 1e6;
    LOG( "Allocated %.2f MB for particle system.", particlesSize );

    int numNodes = m_grid.nodeCount();
    int numParticles = m_particleSystem->size();

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

    SAFE_DELETE( m_hostParticleCache );
    m_hostParticleCache = new ParticleCache;
    cudaMalloc( (void**)&m_hostParticleCache->sigmas, numParticles*sizeof(mat3) );
    cudaMalloc( (void**)&m_hostParticleCache->Aps, numParticles*sizeof(mat3) );
    cudaMalloc( (void**)&m_hostParticleCache->FeHats, numParticles*sizeof(mat3) );
    cudaMalloc( (void**)&m_hostParticleCache->ReHats, numParticles*sizeof(mat3) );
    cudaMalloc( (void**)&m_hostParticleCache->SeHats, numParticles*sizeof(mat3) );
    cudaMalloc( (void**)&m_hostParticleCache->dFs, numParticles*sizeof(mat3) );
    cudaMalloc( (void**)&m_devParticleCache, sizeof(ParticleCache) );
    cudaMemcpy( m_devParticleCache, m_hostParticleCache, sizeof(ParticleCache), cudaMemcpyHostToDevice );
    float particleCachesSize = numParticles*6*sizeof(mat3) / 1e6;
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

    // Free the particle cache using the host structure
    cudaFree( m_hostParticleCache->sigmas );
    cudaFree( m_hostParticleCache->Aps );
    cudaFree( m_hostParticleCache->FeHats );
    cudaFree( m_hostParticleCache->ReHats );
    cudaFree( m_hostParticleCache->SeHats );
    cudaFree( m_hostParticleCache->dFs );
    SAFE_DELETE( m_hostParticleCache );
    cudaFree( m_devParticleCache );

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
