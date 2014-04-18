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

    #include "cuda/vector.cu"
    #include "cuda/matrix.cu"

struct Particle
{
    vec3 position;
    vec3 velocity;
    float mass;
    float volume;
    mat3 elasticF;
    mat3 plasticF;
    Particle() : velocity(.01f, 0.f, 0.f), mass(1e-6), volume(1e-9), elasticF(1.f), plasticF(1.f) {}
};

#ifndef CUDA_INCLUDE

#include <QVector>
typedef unsigned int GLuint;

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

    GLuint vbo() const { return m_glVBO; }

    void merge( const ParticleSystem &particles ) { m_particles += particles.m_particles; deleteVBO(); }

    ParticleSystem& operator += ( const ParticleSystem &particles ) { m_particles += particles.m_particles; deleteVBO(); return *this; }
    ParticleSystem& operator += ( const Particle &particle ) { m_particles.append(particle); deleteVBO(); return *this; }

protected:

    QVector<Particle> m_particles;
    GLuint m_glVBO;

    bool hasVBO() const;
    void buildVBO();
    void deleteVBO();

};

#endif

#endif // PARTICLE_H
