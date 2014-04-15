/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   particle.h
**   Author: mliberma
**   Created: 6 Apr 2014
**
**************************************************************************/

#ifndef PARTICLE_H
#define PARTICLE_H

#ifdef CUDA_INCLUDE
    #include "cuda/vector.cu"
    #include "cuda/matrix.cu"
#else
    #include <glm/vec3.hpp>
    #include <glm/mat3x3.hpp>
#endif

struct Particle
{
#ifdef CUDA_INCLUDE
    typedef vec3 vector_type;
    typedef mat3 matrix_type;
#else
    typedef glm::vec3 vector_type;
    typedef glm::mat3 matrix_type;
#endif

    vector_type position;
    vector_type velocity;
    float mass;
    float volume;
    matrix_type elasticF;
    matrix_type plasticF;
    Particle() : elasticF(1.f), plasticF(1.f) {}
};

#ifndef CUDA_INCLUDE

#include <QVector>
typedef unsigned int GLuint;
struct cudaGraphicsResource;

#include "common/renderable.h"

class ParticleSystem : public Renderable
{

public:

    ParticleSystem();
    virtual ~ParticleSystem();

    void clear();
    inline int size() const { return m_particles.size(); }
    inline void resize( int n ) { m_particles.resize(n); }

    Particle* data() { return m_particles.data(); }
    const QVector<Particle>& getParticles() const { return m_particles; }
    QVector<Particle>& particles() { return m_particles; }

    virtual void render();
    void update( float time );

    ParticleSystem& operator += ( const Particle &particle ) { m_particles.append(particle); return *this; }

private:

    QVector<Particle> m_particles;
    GLuint m_glVBO;
    cudaGraphicsResource *m_cudaVBO;

    bool hasVBO() const;
    void buildVBO();
    void deleteVBO();

};

#endif

#endif // PARTICLE_H
