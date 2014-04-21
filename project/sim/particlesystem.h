/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   particlesystem.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 21 Apr 2014
**
**************************************************************************/

#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include <QVector>

class QGLShaderProgram;
typedef unsigned int GLuint;

#include "common/renderable.h"
#include "geometry/grid.h"
#include "sim/particle.h"

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

    virtual BBox getBBox( const glm::mat4 &ctm );

    GLuint vbo() const { return m_glVBO; }

    void merge( const ParticleSystem &particles ) { m_particles += particles.m_particles; deleteVBO(); }

    ParticleSystem& operator += ( const ParticleSystem &particles ) { m_particles += particles.m_particles; deleteVBO(); return *this; }
    ParticleSystem& operator += ( const Particle &particle ) { m_particles.append(particle); deleteVBO(); return *this; }

protected:

    static QGLShaderProgram *SHADER;
    static QGLShaderProgram* shader();

    QVector<Particle> m_particles;
    GLuint m_glVBO;
    GLuint m_glVAO;

    bool hasVBO() const;
    void buildVBO();
    void deleteVBO();

};

#endif // PARTICLESYSTEM_H
