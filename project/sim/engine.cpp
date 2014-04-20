/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   engine.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#include "common/common.h"
#include "sim/collider.h"
#include "sim/engine.h"
#include "sim/particle.h"
#include "sim/particlegrid.h"

#include "cuda/functions.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <QtConcurrentRun>

#include "GL/gl.h"

#define TICKS 50

Engine::Engine()
    : m_particleSystem( NULL ),
      m_time( 0.f ),
      m_running( false ),
      m_paused( false )
{
    m_particleSystem = new ParticleSystem;

    m_params.timeStep = 0.001f;
    m_params.startTime = 0.f;
    m_params.endTime = 60.f;
    m_params.gravity = vec3( 0.f, -9.8f, 0.f );

    ImplicitCollider collider;
    collider.center = vec3( 0.f, 0.5f, 0.f );
    collider.param = vec3( 0.f, 1.f, 0.f );
    collider.type = HALF_PLANE;
    m_colliders += collider;

    assert( connect(&m_ticker, SIGNAL(timeout()), this, SLOT(update())) );
}

Engine::~Engine()
{
    if ( m_running ) stop();
    SAFE_DELETE( m_particleSystem );
}

void Engine::start()
{
    if ( m_particleSystem->size() > 0 && !m_grid.empty() && !m_running ) {

        LOG( "STARTING SIMULATION" );

        m_time = 0.f;
        initializeCudaResources();
        m_ticker.start(TICKS);
        m_running = true;

//        update();

//        stop();

    } else {
        LOG( "EMPTY PARTICLE SYSTEM OR EMPTY GRID OR SIMULATION ALREADY RUNNING." );
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
    if ( m_paused && m_running ) {
        m_paused = false;
        m_ticker.start(TICKS);
    }
}

void Engine::update()
{
    if ( !m_busy && m_running && !m_paused ) {

        m_busy = true;

        cudaGraphicsMapResources( 1, &m_cudaResource, 0 );
        Particle *devParticles;
        size_t size;
        checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**)&devParticles, &size, m_cudaResource ) );

        if ( (int)(size/sizeof(Particle)) != m_particleSystem->size() ) {
            LOG( "Particle resource error : %lu bytes (%lu expected)", size, m_particleSystem->size()*sizeof(Particle) );
        }

        updateParticles( m_params, devParticles, m_particleSystem->size(), m_devGrid,
                         m_devNodes, m_grid.nodeCount(), m_devPGTD, m_devColliders, m_colliders.size(), m_devMaterial );

//        cudaMemcpy( m_particleSystem->particles().data(), devParticles, 12*sizeof(Particle), cudaMemcpyDeviceToHost );
//        for ( int i = 0; i < 12; ++i ) {
//            Particle &p = m_particleSystem->particles()[i];
//             LOG( "Position: (%f, %f, %f), Velocity: (%f, %f, %f), mass: %g", p.position.x, p.position.y, p.position.z, p.velocity.x, p.velocity.y, p.velocity.z, p.mass );
//        }
//        LOG("\n");

        checkCudaErrors( cudaGraphicsUnmapResources( 1, &m_cudaResource, 0 ) );

        m_time += m_params.timeStep;

        m_busy = false;

    } else {
        LOG( "IS PAUSED OR ISN'T RUNNING" );
    }
}

void Engine::initializeCudaResources()
{
    LOG( "Initializing CUDA resources..." );

    // Particles
    registerVBO( &m_cudaResource, m_particleSystem->vbo() );
    float particlesSize = m_particleSystem->size()*sizeof(Particle) / 1e6;
    LOG( "Allocated %.2f MB for particle system.", particlesSize );

    // Grid
    checkCudaErrors(cudaMalloc( (void**)&m_devGrid, sizeof(Grid) ));
    checkCudaErrors(cudaMemcpy( m_devGrid, &m_grid, sizeof(Grid), cudaMemcpyHostToDevice ));

    // Grid Nodes
    checkCudaErrors(cudaMalloc( (void**)&m_devNodes, m_grid.nodeCount()*sizeof(ParticleGrid::Node) ));
    float nodesSize =  m_grid.nodeCount()*sizeof(ParticleGrid::Node) / 1e6;
    LOG( "Allocating %.2f MB for grid nodes.", nodesSize );

    // Colliders
    checkCudaErrors(cudaMalloc( (void**)&m_devColliders, m_colliders.size()*sizeof(ImplicitCollider) ));
    checkCudaErrors(cudaMemcpy( m_devColliders, m_colliders.data(), m_colliders.size()*sizeof(ImplicitCollider), cudaMemcpyHostToDevice ));

    // Particle Grid Temp Data
    checkCudaErrors(cudaMalloc( (void**)&m_devPGTD, m_particleSystem->size()*sizeof(ParticleGridTempData) ));
    float tempSize = m_particleSystem->size()*sizeof(ParticleGridTempData) / 1e6;
    LOG( "Allocating %.2f MB for particle grid temp data.", tempSize );

    // Material Constants
    checkCudaErrors(cudaMalloc( (void**)&m_devMaterial, sizeof(MaterialConstants) ));
    checkCudaErrors(cudaMemcpy( m_devMaterial, &m_materialConstants, sizeof(MaterialConstants), cudaMemcpyHostToDevice ));

    LOG("Allocated %.2f MB in total", particlesSize + nodesSize + tempSize );
}

void Engine::freeCudaResources()
{
    unregisterVBO( m_cudaResource );
    cudaFree( m_devGrid );
    cudaFree( m_devNodes );
    cudaFree( m_devColliders );
    cudaFree( m_devPGTD );
    cudaFree( m_devMaterial );
}

