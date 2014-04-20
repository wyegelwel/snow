/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   camera.h
**   Author: mliberma
**   Created: 6 Apr 2014
**
**************************************************************************/

#ifndef CAMERA_H
#define CAMERA_H

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/vec3.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"

class Camera
{

public:

    Camera()
    {
        m_aspect = 1.f;
        m_near = 0.01f;
        m_far = 1e6;
        m_heightAngle = M_PI/3.f;
        updateProjectionMatrix();
        orient( glm::vec3(1, 1, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0) );
    }

    void orient( const glm::vec3 &eye, const glm::vec3 &lookAt, const glm::vec3 &up )
    {
        m_eye = eye;
        m_lookAt = lookAt;
        m_look = glm::normalize(m_lookAt-m_eye);
        m_up = up;
        m_w = -m_look;
        m_v = glm::normalize(m_up - (glm::dot(m_up,m_w)*m_w));
        m_u = glm::cross(m_v, m_w);
        updateModelviewMatrix();
    }

    glm::mat4 getModelviewMatrix() const { return m_modelview; }
    glm::mat4 getProjectionMatrix() const { return m_projection; }

    glm::vec3 getPosition() const { return m_eye; }
    glm::vec3 getLookAt() const { return m_lookAt; }
    glm::vec3 getLook() const { return m_look; }
    glm::vec3 getUp() const { return m_up; }
    glm::vec3 getU() const { return m_u; }
    glm::vec3 getV() const { return m_v; }
    glm::vec3 getW() const { return m_w; }

    float getAspect() const { return m_aspect; }
    void setAspect( float aspect ) { m_aspect = aspect; updateProjectionMatrix(); }

    float getNear() const { return m_near; }
    float getFar() const { return m_far; }
    void setClip( float near, float far ) { m_near = near; m_far = far; updateProjectionMatrix(); }

    float getHeightAngle() const { return m_heightAngle; }
    void setHeightAngle( float radians ) { m_heightAngle = radians; updateProjectionMatrix(); }

    float getFocusDistance() const { return glm::length(m_lookAt-m_eye); }

    glm::vec3 getCameraRay( const glm::vec2 &uv ) const
    {
        glm::vec3 camDir = glm::vec3(2.f*uv.x-1.f,1.f-2.f*uv.y,-1.f/tanf(m_heightAngle/2.f));
        glm::vec3 worldDir = m_aspect*camDir.x*m_u + camDir.y*m_v + camDir.z*m_w;
        return glm::normalize(worldDir);
    }

private:

    glm::mat4 m_modelview, m_projection;
    glm::vec3 m_eye, m_lookAt;
    glm::vec3 m_look, m_up, m_u, m_v, m_w;
    float m_aspect, m_near, m_far;
    float m_heightAngle;

    void updateModelviewMatrix()
    {
        glm::mat4 translation = glm::translate(glm::mat4(1.f), -m_eye);
        glm::mat4 rotation = glm::mat4( m_u.x, m_v.x, m_w.x, 0,
                                        m_u.y, m_v.y, m_w.y, 0,
                                        m_u.z, m_v.z, m_w.z, 0,
                                            0,     0,     0, 1 );
        m_modelview = rotation * translation;
    }

    void updateProjectionMatrix()
    {
        float tanH = tanf(m_heightAngle/2.f);
        float tanW = m_aspect*tanH;
        glm::mat4 normalizing = glm::scale(glm::mat4(1.f), glm::vec3(1.f/(m_far*tanW), 1.f/(m_far*tanH), 1.f/m_far));
        float c = -m_near/m_far;
        glm::mat4 unhinging = glm::mat4( 1, 0,        0,  0,
                                         0, 1,        0,  0,
                                         0, 0, -1/(c+1), -1,
                                         0, 0,  c/(c+1),  0 );
        m_projection = unhinging * normalizing;
    }

};

#endif // CAMERA_H
