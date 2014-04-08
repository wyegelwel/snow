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

#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

struct Particle
{
    glm::vec3 position;
    glm::vec3 velocity;
    float mass;
    float volume;
    glm::mat3 elasticF;
    glm::mat3 plasticF;
#ifndef CUDA_INCLUDE
    Particle() : elasticF(1.f), plasticF(1.f) {}
#endif
};

#ifndef CUDA_INCLUDE

#include <QVector>
typedef unsigned int GLuint;
struct cudaGraphicsResource;

#include "scene/renderable.h"

class ParticleSystem : public Renderable
{

public:

    ParticleSystem();
    virtual ~ParticleSystem();

    void clear();

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
