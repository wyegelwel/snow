/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   viewport.h
**   Author: mliberma
**   Created: 6 Apr 2014
**
**************************************************************************/

#ifndef VIEWPORT_H
#define VIEWPORT_H

#include <glm/vec3.hpp>

class QWidget;
class Camera;

class Viewport
{

public:

    Viewport();
    ~Viewport();

    Camera* getCamera() const { return m_camera; }

    void loadMatrices() const;
    void popMatrices() const;

    void push() const;
    void pop() const;

    void orient( const glm::vec3 &eye, const glm::vec3 &lookAt, const glm::vec3 &up );

    void setDimensions( int width, int height );

private:

    Camera *m_camera;
    int m_width, m_height;

};

#endif // VIEWPORT_H
