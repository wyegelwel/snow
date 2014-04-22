/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   movetool.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 20 Apr 2014
**
**************************************************************************/

#include <GL/gl.h>

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "movetool.h"

#include "common/common.h"
#include "scene/scene.h"
#include "scene/scenenode.h"
#include "scene/scenenodeiterator.h"
#include "ui/picker.h"
#include "ui/uisettings.h"
#include "ui/userinput.h"
#include "ui/viewpanel.h"
#include "viewport/camera.h"
#include "viewport/viewport.h"

MoveTool::MoveTool( ViewPanel *panel )
    : SelectionTool(panel),
      m_axisSelection(Picker::NO_PICK),
      m_active(false),
      m_moving(false),
      m_center(0,0,0),
      m_vbo(0)
{
}

MoveTool::~MoveTool()
{
    deleteVBO();
}

void
MoveTool::update()
{
    if ( (m_active = SelectionTool::hasSelection()) ) {
        m_center = SelectionTool::getSelectionCenter();
    }
}

void
MoveTool::renderAxis( unsigned int i ) const
{
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glm::mat4 translate = glm::translate( glm::mat4(1.f), glm::vec3(m_center.x, m_center.y, m_center.z) );
    glMultMatrixf( glm::value_ptr(translate*Tool::axialBasis(i)) );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, sizeof(vec3), (void*)(0) );
    glLineWidth( 2.f );
    glDrawArrays( GL_LINES, 0, 2 );
    glDrawArrays( GL_TRIANGLES, 2, m_vboSize-2 );
    glDisableClientState( GL_VERTEX_ARRAY );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glPopMatrix();
}

void
MoveTool::render()
{
    if ( m_active ) {

        if ( !hasVBO() ) buildVBO();

        glPushAttrib( GL_DEPTH_BUFFER_BIT );
        glDisable( GL_DEPTH_TEST );
        glPushAttrib( GL_LIGHTING_BIT );
        glDisable( GL_LIGHTING );
        glPushAttrib( GL_COLOR_BUFFER_BIT );
        glEnable( GL_BLEND );
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        glEnable( GL_LINE_SMOOTH );
        glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
        for ( unsigned int i = 0; i < 3; ++i ) {
            vec3 color( 0, 0, 0 );
            if ( m_axisSelection == i ) color = vec3( 1, 1, 0 );
            else color[i] = 1;
            glColor3fv( color.data );
            renderAxis( i );
        }
        glPopAttrib();
        glPopAttrib();
        glPopAttrib();
    }
}

unsigned int
MoveTool::getAxisPick() const
{
    unsigned int pick = Picker::NO_PICK;
    if ( m_active ) {
        m_panel->m_viewport->loadPickMatrices( UserInput::mousePos(), 6.f );
        Picker picker( 3 );
        for ( unsigned int i = 0; i < 3; ++i ) {
            picker.setObjectIndex( i );
            renderAxis( i );
        }
        pick = picker.getPick();
        m_panel->m_viewport->popMatrices();
    }
    return pick;
}

void
MoveTool::mousePressed()
{
    if ( m_active ) {
        m_axisSelection = getAxisPick();
        m_moving = ( m_axisSelection != Picker::NO_PICK );
        if ( m_axisSelection == Picker::NO_PICK ) {
            SelectionTool::mousePressed();
        }
    } else {
        SelectionTool::mousePressed();
    }
    update();
}

vec3
MoveTool::intersectAxis( const glm::ivec2 &mouse ) const
{
    glm::vec2 uv = glm::vec2( (float)mouse.x/m_panel->width(), (float)mouse.y/m_panel->height() );
    vec3 direction = m_panel->m_viewport->getCamera()->getCameraRay( uv );
    vec3 origin = m_panel->m_viewport->getCamera()->getPosition();
    unsigned int majorAxis = direction.majorAxis();
    int axis = majorAxis;
    if ( majorAxis == m_axisSelection ) {
        axis = ( majorAxis == 0 ) ? 1 : 0;
    }
    float t = (m_center[axis]-origin[axis])/direction[axis];
    vec3 point = origin + t*direction;
    vec3 a = vec3(0,0,0); a[m_axisSelection] = 1.f;
    float d = vec3::dot( a, point-m_center );
    return m_center + d*a;
}

void
MoveTool::mouseMoved()
{
    if ( m_moving ) {
        vec3 p0 = intersectAxis( UserInput::mousePos()-UserInput::mouseMove() );
        vec3 p1 = intersectAxis( UserInput::mousePos() );
        vec3 t = p1-p0;
        glm::vec3 translate = glm::vec3( t.x, t.y, t.z );
        glm::mat4 transform = glm::translate( glm::mat4(1.f), translate );
        for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
            if ( (*it)->hasRenderable() && (*it)->getRenderable()->isSelected() ) {
                (*it)->applyTransformation( transform );
            }
        }
    }
    update();
}

void
MoveTool::mouseReleased()
{
    m_axisSelection = Picker::NO_PICK;
    m_moving = false;
    SelectionTool::mouseReleased();
    update();
}

bool
MoveTool::hasVBO() const
{
    return m_vbo > 0 && glIsBuffer( m_vbo );
}

void
MoveTool::buildVBO()
{
    deleteVBO();

    QVector<vec3> data;
    data += vec3( 0, 0, 0 );
    data += vec3( 0, 1, 0 );

    static const int resolution = 60;
    static const float dTheta = 2.f*M_PI/resolution;
    static const float coneHeight = 0.1f;
    static const float coneRadius = 0.05f;
    for ( int i = 0; i < resolution; ++i ) {
        data += vec3( 0, 1, 0 );
        float theta0 = i*dTheta;
        float theta1 = (i+1)*dTheta;
        data += (vec3(0,1-coneHeight,0)+coneRadius*vec3(cosf(theta0),0,-sinf(theta0)));
        data += (vec3(0,1-coneHeight,0)+coneRadius*vec3(cosf(theta1),0,-sinf(theta1)));
    }

    glGenBuffers( 1, &m_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glBufferData( GL_ARRAY_BUFFER, data.size()*sizeof(vec3), data.data(), GL_STATIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    m_vboSize = data.size();
}

void
MoveTool::deleteVBO()
{
    if ( m_vbo > 0 ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
        if ( glIsBuffer(m_vbo) ) glDeleteBuffers( 1, &m_vbo );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        m_vbo = 0;
    }
}
