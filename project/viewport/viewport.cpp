/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   viewport.cpp
**   Author: mliberma
**   Created: 6 Apr 2014
**
**************************************************************************/

#include "viewport.h"

#include <GL/gl.h>
#include <glm/gtc/type_ptr.hpp>

#include "common/common.h"
#include "viewport/camera.h"

Viewport::Viewport()
{
    m_camera = new Camera;
    m_width = 1000;
    m_height = 1000;
    m_camera->setClip( 0.01f, 1000.f );
    m_camera->setHeightAngle( M_PI/3.f );
}

Viewport::~Viewport()
{
    SAFE_DELETE( m_camera );
}

void
Viewport::loadMatrices() const
{
    glm::mat4 modelview = m_camera->getModelviewMatrix();
    glm::mat4 projection = m_camera->getProjectionMatrix();
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadMatrixf( glm::value_ptr(modelview) );
    glMatrixMode( GL_PROJECTION );
    glPushMatrix();
    glLoadMatrixf( glm::value_ptr(projection) );
}

void
Viewport::popMatrices() const
{
    glMatrixMode( GL_MODELVIEW );
    glPopMatrix();
    glMatrixMode( GL_PROJECTION );
    glPopMatrix();
}

void
Viewport::push() const
{
    glViewport( 0, 0, m_width, m_height );
    loadMatrices();
}

void
Viewport::pop() const
{
    popMatrices();
}

void
Viewport::orient( const glm::vec3 &eye,
                  const glm::vec3 &lookAt,
                  const glm::vec3 &up )
{
    m_camera->orient(eye, lookAt, up);
}

void
Viewport::setDimensions( int width, int height )
{
    m_camera->setAspect( float(width)/float(height) );
    m_width = width;
    m_height = height;
}
