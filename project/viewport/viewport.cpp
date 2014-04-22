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

#include "common/common.h"
#include "ui/userinput.h"
#include "ui/tools/tool.h"
#include "viewport/camera.h"

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/gtc/type_ptr.hpp"

#include "cuda/vector.cu"

#define ZOOM_SCALE 1.f

Viewport::Viewport()
{
    m_camera = new Camera;
    m_width = 1000;
    m_height = 1000;
    m_camera->setClip( 0.01f, 1000.f );
    m_camera->setHeightAngle( M_PI/6.f );
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
Viewport::popMatrices()
{
    glMatrixMode( GL_MODELVIEW );
    glPopMatrix();
    glMatrixMode( GL_PROJECTION );
    glPopMatrix();
}

void
Viewport::loadPickMatrices( const glm::ivec2 &click, float size ) const
{
    const glm::mat4 &modelview = m_camera->getModelviewMatrix();
    const glm::mat4 &projection = m_camera->getProjectionMatrix();

    glMatrixMode( GL_PROJECTION );
    glPushMatrix();
    glLoadIdentity();
    float width = (float)m_width;
    float height = (float)m_height;
    float tX = 2.f*(width/2.f-click.x)/size;
    float tY = 2.f*(click.y-height/2.f+1.f)/size;
    const glm::mat4 translate = glm::translate( glm::mat4(1.f), glm::vec3(tX,tY,0.f) );
    glMultMatrixf( glm::value_ptr(translate) );
    const glm::mat4 scale = glm::scale( glm::mat4(1.f), glm::vec3(width/size, height/size, 1.f) );
    glMultMatrixf( glm::value_ptr(scale) );
    glMultMatrixf( glm::value_ptr(projection) );

    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadMatrixf( glm::value_ptr(modelview) );
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

void
Viewport::mouseMoved()
{
    glm::ivec2 pos = UserInput::mousePos();
    glm::vec2 posf = glm::vec2( pos.x/float(m_width), pos.y/float(m_height) );
    glm::ivec2 move = UserInput::mouseMove();
    glm::vec2 movef = glm::vec2( move.x/float(m_width), move.y/float(m_height) );

    switch ( m_state ) {
    case IDLE:
        break;
    case PANNING:
    {
        float tanH = tanf(m_camera->getHeightAngle()/2.f);
        float tanW = m_camera->getAspect()*tanH;
        float du = -2.f*movef.x*m_camera->getFocusDistance()*tanW;
        float dv = 2.f*movef.y*m_camera->getFocusDistance()*tanH;
        glm::vec3 trans = du*m_camera->getU() + dv*m_camera->getV();
        m_camera->orient( m_camera->getPosition()+trans, m_camera->getLookAt()+trans, m_camera->getUp() );
        break;
    }
    case ZOOMING:
    {
        float focus = m_camera->getFocusDistance();
        if ( movef.x < 0 && fabsf(movef.x) > focus ) break;
        glm::vec3 trans = focus*ZOOM_SCALE*movef.x*m_camera->getW();
        m_camera->orient( m_camera->getPosition()+trans, m_camera->getLookAt(), m_camera->getUp() );
        break;
    }
    case TUMBLING:
    {
        float ax = movef.x * 1.5f * M_PI;
        float ay = movef.y * 1.5f * M_PI;
        float alpha = (posf.x-movef.x/2.f-0.5f)/0.5f;
        float yaw = (1.f - fabsf(alpha)) * ay;
        float pitch = ax;
        float roll = alpha * ay;
        glm::vec4 eye = glm::vec4( m_camera->getPosition(), 1.f );
        glm::vec4 lookAt = glm::vec4( m_camera->getLookAt(), 1.f );
        glm::mat4 T = glm::translate( glm::mat4(1.f), -glm::vec3(eye.x, eye.y, eye.z) );
        glm::mat4 Tinv = glm::translate( glm::mat4(1.f), glm::vec3(eye.x, eye.y, eye.z) );
        glm::mat4 RV = glm::rotate( glm::mat4(1.f), -pitch, m_camera->getV() );
        glm::mat4 RU = glm::rotate( glm::mat4(1.f), -yaw, m_camera->getU() );
        glm::mat4 RW = glm::rotate( glm::mat4(1.f), roll, m_camera->getW() );
        eye = lookAt + (Tinv*RW*RU*RV*T)*(eye-lookAt);
        glm::vec4 up = RW*RU*RV*glm::vec4( m_camera->getUp(), 0.f );
        m_camera->orient( glm::vec3(eye), glm::vec3(lookAt), glm::vec3(up) );
        break;
    }
    }
}

void
Viewport::drawAxis()
{
    static const float corner = 50.f;
    static const float distance = 10.f;
    glm::vec2 uv = glm::vec2( corner/m_width, 1.f-(corner/m_height) );
    glm::vec3 c = m_camera->getPosition() + distance*m_camera->getCameraRay( uv );
    static const float length = 0.25f;
    glm::vec3 x = c + length*glm::vec3(1,0,0);
    glm::vec3 y = c + length*glm::vec3(0,1,0);
    glm::vec3 z = c + length*glm::vec3(0,0,1);

    glPushAttrib( GL_DEPTH_BUFFER_BIT );
    glDisable( GL_DEPTH_TEST );
    glPushAttrib( GL_COLOR_BUFFER_BIT );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_LINE_SMOOTH );
    glHint( GL_LINE_SMOOTH, GL_NICEST );
    glLineWidth( 1.5f );
    glBegin( GL_LINES ); {
        glColor3fv( Tool::getAxialColor(0).data );
        glVertex3f( c.x, c.y, c.z );
        glVertex3f( x.x, x.y, x.z );
        glColor3fv( Tool::getAxialColor(1).data );
        glVertex3f( c.x, c.y, c.z );
        glVertex3f( y.x, y.y, y.z );
        glColor3fv( Tool::getAxialColor(2).data );
        glVertex3f( c.x, c.y, c.z );
        glVertex3f( z.x, z.y, z.z );
    } glEnd();
    glPopAttrib();
    glPopAttrib();
}
