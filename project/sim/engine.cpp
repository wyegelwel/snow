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
#include "sim/caches.h"
#include "sim/collider.h"
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

#define TICKS 50

Engine::Engine()
    : m_particleSystem(NULL),
      m_particleGrid(NULL),
      m_time(0.f),
      m_running(false),
      m_paused(false)
{
    m_particleSystem = new ParticleSystem;
    m_particleGrid =  new ParticleGrid;

    m_exporter = NULL;

    m_params.timeStep = 5e-5;
    m_params.startTime = 0.f;
    m_params.endTime = 60.f;
    m_params.gravity = vec3( 0.f, -9.8f, 0.f );

    assert( connect(&m_ticker, SIGNAL(timeout()), this, SLOT(update())) );
}

Engine::~Engine()
{
    if ( m_running ) stop();
    SAFE_DELETE( m_particleSystem );
    SAFE_DELETE( m_particleGrid );
    if ( m_export )
        SAFE_DELETE( m_exporter );
}

void Engine::setGrid(const Grid &grid)
{
    m_grid = grid;
    m_particleGrid->setGrid( grid );
}

void Engine::addCollider(Collider &collider, const glm::mat4 &ctm)  {
    ImplicitCollider &col = *(collider.getImplicitCollider());
//    glm::vec4 centerTemp = glm::vec4(col.center.x,col.center.y,col.center.z,1);
//    //transform center
//    centerTemp = ctm*centerTemp;
//    col.center = glm::vec3(centerTemp.x,centerTemp.y,centerTemp.z);

    //transform center
    std::cout << "ctm: " << std::endl << ctm[0][0] << "," << ctm[0][1] << "," << ctm[0][2] << "," << ctm[0][3] << std::endl
                         << std::endl << ctm[1][0] << "," << ctm[1][1] << "," << ctm[1][2] << "," << ctm[1][3] << std::endl
                         << std::endl << ctm[2][0] << "," << ctm[2][1] << "," << ctm[2][2] << "," << ctm[2][3] << std::endl
                         << std::endl << ctm[3][0] << "," << ctm[3][1] << "," << ctm[3][2] << "," << ctm[3][3] << std::endl << std::endl;
    col.center = glm::vec3(ctm[3][0],ctm[3][1],ctm[3][2]);
    std::cout << "center: " << col.center.x << "," << col.center.y << "," << col.center.z << std::endl;
    //transform scale for sphere
    glm::vec3 scale(glm::length(glm::vec4(ctm[0][0],ctm[0][1],ctm[0][2],ctm[0][3])),glm::length(glm::vec4(ctm[1][0],ctm[1][1],ctm[1][2],ctm[1][3])),glm::length(glm::vec4(ctm[2][0],ctm[2][1],ctm[2][2],ctm[2][3])));
    if(col.type == SPHERE)  {
        col.param.x = scale.x;
        std::cout << "param: " << col.param.x << std::endl;
    }

    if(col.type == HALF_PLANE)  {
        glm::vec4 basisOne(glm::vec4(ctm[0][0],ctm[0][1],ctm[0][2],ctm[0][3])/scale.x);
        glm::vec4 basisTwo(glm::vec4(ctm[1][0],ctm[1][1],ctm[1][2],ctm[1][3])/scale.x);
        glm::vec4 basisThree(glm::vec4(ctm[2][0],ctm[2][1],ctm[2][2],ctm[2][3])/scale.x);
        glm::mat3 rotation(basisOne.x,basisTwo.x,basisThree.x,
                           basisOne.y,basisTwo.y,basisThree.y,
                           basisOne.z,basisTwo.z,basisThree.z);
        col.param = glm::vec3(0,1,0)*rotation;
        std::cout << "param: " << col.param.x << "," << col.param.y << "," << col.param.z << std::endl;
    }
    m_colliders.append(col);
}

void Engine::addParticleSystem( const ParticleSystem &particles )
{
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

bool Engine::start( bool exportScene )
{
    if ( m_particleSystem->size() > 0 && !m_grid.empty() && !m_running ) {

        if ( (m_export = exportScene) ) m_exporter->reset( m_grid );

        LOG( "STARTING SIMULATION" );

        m_time = 0.f;
        m_params.timeStep = UiSettings::timeStep();
        initializeCudaResources();
        m_running = true;
        m_ticker.start(TICKS);

        return true;

//        update();
//        stop();
//        return false;

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

        bool doShading = UiSettings::showParticlesMode() == UiSettings::PARTICLE_SHADED;
        updateParticles( m_params, devParticles, m_devParticleCaches, m_particleSystem->size(), m_devGrid,
                         devNodes, m_devNodeCaches, m_grid.nodeCount(), m_devColliders, m_colliders.size(), m_devMaterial, doShading );

        if (m_export && (m_time - m_exporter->getLastUpdateTime() >= m_exporter->getspf()))
        {
            cudaMemcpy(m_exporter->getNodesPtr(), devNodes, m_grid.nodeCount() * sizeof(Node), cudaMemcpyDeviceToHost);
            m_exporter->runExportThread(m_time);
        }

        checkCudaErrors( cudaGraphicsUnmapResources( 1, &m_particlesResource, 0 ) );
        checkCudaErrors( cudaGraphicsUnmapResources( 1, &m_nodesResource, 0 ) );
        checkCudaErrors( cudaDeviceSynchronize() );

        m_time += m_params.timeStep;
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

    // Material Constants
    checkCudaErrors(cudaMalloc( (void**)&m_devMaterial, sizeof(MaterialConstants) ));
    checkCudaErrors(cudaMemcpy( m_devMaterial, &m_materialConstants, sizeof(MaterialConstants), cudaMemcpyHostToDevice ));

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
