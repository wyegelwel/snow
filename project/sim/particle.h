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

#include <QVector>
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

typedef unsigned int GLuint;

struct cudaGraphicsResource;

struct Particle
{
    glm::vec3 position;
    glm::vec3 velocity;
    float mass;
    float volume;
    glm::mat3 elasticF;
    glm::mat3 plasticF;
    Particle() : elasticF(1.f), plasticF(1.f) {}
};

class ParticleSystem
{

public:

    ParticleSystem();
    ~ParticleSystem();

    void clear();

    const QVector<Particle>& getParticles() const { return m_particles; }
    QVector<Particle>& particles() { return m_particles; }

    void render();
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

#endif // PARTICLE_H
