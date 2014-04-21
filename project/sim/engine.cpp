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
#include "sim/collider.h"
#include "sim/engine.h"
#include "sim/griddataviewer.h"
#include "sim/particle.h"
#include "sim/particlegridnode.h"
#include "ui/uisettings.h"

#include "cuda/functions.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define TICKS 50

Engine::Engine()
    : m_particleSystem( NULL ),
      m_time( 0.f ),
      m_running( false ),
      m_paused( false )
{
    m_gridViewer = NULL;

    m_particleSystem = new ParticleSystem;

    m_exporter = NULL;

    m_params.timeStep = 0.001f;
    m_params.startTime = 0.f;
    m_params.endTime = 60.f;
    m_params.gravity = vec3( 0.f, -9.8f, 0.f );

//    ImplicitCollider collider;
//    collider.center = vec3( 0.f, 0.5f, 0.f );
//    collider.param = vec3( 0.f, 1.f, 0.f );
//    collider.type = HALF_PLANE;
//    m_colliders += collider;

    assert( connect(&m_ticker, SIGNAL(timeout()), this, SLOT(update())) );
}

Engine::~Engine()
{
    if ( m_running ) stop();
    SAFE_DELETE( m_particleSystem );
    if ( m_export )
        SAFE_DELETE(m_exporter);
    SAFE_DELETE( m_gridViewer );
}

void Engine::addParticleSystem( const ParticleSystem &particles )
{
    *m_particleSystem += particles;
}

void Engine::clearParticleSystem()
{
    m_particleSystem->clear();
}

void Engine::initExporter( QString fprefix )
{
    m_exporter = new MitsubaExporter( fprefix, 24.f );
}

void Engine::start( bool exportScene )
{
    if ( m_particleSystem->size() > 0 && !m_grid.empty() && !m_running ) {

        if ( (m_export = exportScene) ) m_exporter->reset( m_grid );

        LOG( "STARTING SIMULATION" );

        SAFE_DELETE( m_gridViewer );
        m_gridViewer = new GridDataViewer( m_grid );

        m_time = 0.f;
        initializeCudaResources();
        m_running = true;
        m_ticker.start(TICKS);

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

bool Engine::isRunning()
{
    return m_running;
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

        if ( UiSettings::showGridData() ) {
            cudaMemcpy( m_gridViewer->data(), m_devNodes, m_gridViewer->byteCount(), cudaMemcpyDeviceToHost );
            m_gridViewer->update();
        }

        if (m_export && (m_time - m_exporter->getLastUpdateTime() >= m_exporter->getspf()))
        {
            // TODO - memcpy the mass data from each ParticleGrid::Node
            // to the m_densities array in the exporter.
            cudaMemcpy(m_exporter->getNodesPtr(), m_devNodes, m_grid.nodeCount() * sizeof(ParticleGridNode), cudaMemcpyDeviceToHost);
            // TODO - call this in a separate thread so that the simulation isn't slowed down while
            // once that is done, call the export function on a separate thread
            // so the rest of the simulation can continue
            //m_exporter->test(m_time);
            m_exporter->exportVolumeData(m_time);
        }


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
    registerVBO( &m_cudaResource, m_particleSystem->vbo() );
    float particlesSize = m_particleSystem->size()*sizeof(Particle) / 1e6;
    LOG( "Allocated %.2f MB for particle system.", particlesSize );

    // Grid
    checkCudaErrors(cudaMalloc( (void**)&m_devGrid, sizeof(Grid) ));
    checkCudaErrors(cudaMemcpy( m_devGrid, &m_grid, sizeof(Grid), cudaMemcpyHostToDevice ));

    // Grid Nodes
    checkCudaErrors(cudaMalloc( (void**)&m_devNodes, m_grid.nodeCount()*sizeof(ParticleGridNode) ));
    float nodesSize =  m_grid.nodeCount()*sizeof(ParticleGridNode) / 1e6;
    LOG( "Allocating %.2f MB for grid nodes.", nodesSize );

    // Colliders
    checkCudaErrors(cudaMalloc( (void**)&m_devColliders, m_colliders.size()*sizeof(ImplicitCollider) ));
    checkCudaErrors(cudaMemcpy( m_devColliders, m_colliders.data(), m_colliders.size()*sizeof(ImplicitCollider), cudaMemcpyHostToDevice ));

    // Particle Grid Temp Data
    checkCudaErrors(cudaMalloc( (void**)&m_devPGTD, m_particleSystem->size()*sizeof(ParticleTempData) ));
    float tempSize = m_particleSystem->size()*sizeof(ParticleTempData) / 1e6;
    LOG( "Allocating %.2f MB for particle grid temp data.", tempSize );

    // Material Constants
    checkCudaErrors(cudaMalloc( (void**)&m_devMaterial, sizeof(MaterialConstants) ));
    checkCudaErrors(cudaMemcpy( m_devMaterial, &m_materialConstants, sizeof(MaterialConstants), cudaMemcpyHostToDevice ));

    LOG("Allocated %.2f MB in total", particlesSize + nodesSize + tempSize );
}

void Engine::freeCudaResources()
{
    LOG( "Freeing CUDA resources..." );
    unregisterVBO( m_cudaResource );
    cudaFree( m_devGrid );
    cudaFree( m_devNodes );
    cudaFree( m_devColliders );
    cudaFree( m_devPGTD );
    cudaFree( m_devMaterial );
}

void Engine::render()
{
    if ( UiSettings::showParticles() ) m_particleSystem->render();
    if ( m_gridViewer && UiSettings::showGridData() ) m_gridViewer->render();
}

BBox Engine::getBBox( const glm::mat4 &ctm )
{
    return m_particleSystem->getBBox( ctm );
}
