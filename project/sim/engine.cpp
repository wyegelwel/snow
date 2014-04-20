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
#include "sim/particle.h"
#include "sim/particlegridnode.h"
#include "ui/uisettings.h"

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
      m_gridVBO(0),
      m_vboSize(0),
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
    deleteVBO();
}

void Engine::addParticleSystem( const ParticleSystem &particles )
{
    *m_particleSystem += particles;
}

void Engine::clearParticleSystem()
{
    m_particleSystem->clear();
}

void Engine::start()
{
    if ( m_particleSystem->size() > 0 && !m_grid.empty() && !m_running ) {

        LOG( "STARTING SIMULATION" );

        m_time = 0.f;
        initializeCudaResources();
        m_ticker.start(TICKS);
        m_running = true;

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
    if ( !hasVBO() ) buildVBO();

    m_particleSystem->render();

    if ( m_vboSize > 0 && UiSettings::showBBox() ) {

        glPushAttrib( GL_COLOR_BUFFER_BIT );
        glEnable( GL_BLEND );
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

        glEnable( GL_LINE_SMOOTH );
        glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );

        glBindBuffer( GL_ARRAY_BUFFER, m_gridVBO );
        glEnableClientState( GL_VERTEX_ARRAY );
        glVertexPointer( 3, GL_FLOAT, sizeof(vec3), (void*)(0) );

        glColor4f( 0.5f, 0.8f, 1.f, 0.75f );
        glLineWidth( 3.f );
        glDrawArrays( GL_LINES, 0, 24 );

        if ( UiSettings::showGrid() ) {
            glColor4f( 0.5f, 0.8f, 1.f, 0.25f );
            glLineWidth( 1.5f );
            glDrawArrays( GL_LINES, 24, m_vboSize-24 );
        }

        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        glDisableClientState( GL_VERTEX_ARRAY );

        glPopAttrib();
    }

}

bool Engine::hasVBO() const
{
    return m_gridVBO > 0 && glIsBuffer( m_gridVBO );
}

void Engine::buildVBO()
{
    deleteVBO();

    QVector<vec3> data;

    const glm::ivec3 &dim = m_grid.dim;
    vec3 min = m_grid.pos;
    vec3 max = m_grid.pos + m_grid.h * vec3( dim.x, dim.y, dim.z );

    // Bounding box
    data += min;
    data += vec3( min.x, min.y, max.z );
    data += vec3( min.x, min.y, max.z );
    data += vec3( min.x, max.y, max.z );
    data += vec3( min.x, max.y, max.z );
    data += vec3( min.x, max.y, min.z );
    data += vec3( min.x, max.y, min.z );
    data += min;
    data += vec3( max.x, min.y, min.z );
    data += vec3( max.x, min.y, max.z );
    data += vec3( max.x, min.y, max.z );
    data += vec3( max.x, max.y, max.z );
    data += vec3( max.x, max.y, max.z );
    data += vec3( max.x, max.y, min.z );
    data += vec3( max.x, max.y, min.z );
    data += vec3( max.x, min.y, min.z );
    data += min;
    data += vec3( max.x, min.y, min.z );
    data += vec3( min.x, min.y, max.z );
    data += vec3( max.x, min.y, max.z );
    data += vec3( min.x, max.y, max.z );
    data += max;
    data += vec3( min.x, max.y, min.z );
    data += vec3( max.x, max.y, min.z );

    // yz faces
    for ( int i = 1; i < dim.y; ++i ) {
        float y = min.y + i*m_grid.h;
        data += vec3( min.x, y, min.z );
        data += vec3( min.x, y, max.z );
        data += vec3( max.x, y, min.z );
        data += vec3( max.x, y, max.z );
    }
    for ( int i = 1; i < dim.z; ++i ) {
        float z = min.z + i*m_grid.h;
        data += vec3( min.x, min.y, z );
        data += vec3( min.x, max.y, z );
        data += vec3( max.x, min.y, z );
        data += vec3( max.x, max.y, z );
    }

    // xy faces
    for ( int i = 1; i < dim.x; ++i ) {
        float x = min.x + i*m_grid.h;
        data += vec3( x, min.y, min.z );
        data += vec3( x, max.y, min.z );
        data += vec3( x, min.y, max.z );
        data += vec3( x, max.y, max.z );
    }
    for ( int i = 1; i < dim.y; ++i ) {
        float y = min.y + i*m_grid.h;
        data += vec3( min.x, y, min.z );
        data += vec3( max.x, y, min.z );
        data += vec3( min.x, y, max.z );
        data += vec3( max.x, y, max.z );
    }

    // xz faces
    for ( int i = 1; i < dim.x; ++i ) {
        float x = min.x + i*m_grid.h;
        data += vec3( x, min.y, min.z );
        data += vec3( x, min.y, max.z );
        data += vec3( x, max.y, min.z );
        data += vec3( x, max.y, max.z );
    }
    for ( int i = 1; i < dim.z; ++i ) {
        float z = min.z + i*m_grid.h;
        data += vec3( min.x, min.y, z );
        data += vec3( max.x, min.y, z );
        data += vec3( min.x, max.y, z );
        data += vec3( max.x, max.y, z );
    }

    m_vboSize = data.size();

    glGenBuffers( 1, &m_gridVBO );
    glBindBuffer( GL_ARRAY_BUFFER, m_gridVBO );
    glBufferData( GL_ARRAY_BUFFER, data.size()*sizeof(vec3), data.data(), GL_STATIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
}

void Engine::deleteVBO()
{
    if ( m_gridVBO > 0 ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_gridVBO );
        if ( glIsBuffer(m_gridVBO) ) glDeleteBuffers( 1, &m_gridVBO );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        m_gridVBO = 0;
    }
}
