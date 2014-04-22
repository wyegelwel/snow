/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scaletool.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 22 Apr 2014
**
**************************************************************************/

#include <GL/gl.h>

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "scaletool.h"

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

#define SCALE 0.01f

ScaleTool::ScaleTool( ViewPanel *panel )
    : SelectionTool(panel),
      m_axisSelection(Picker::NO_PICK),
      m_active(false),
      m_scaling(false),
      m_center(0,0,0),
      m_mouseDownPos(0,0),
      m_transformInverse(1.f),
      m_transform(1.f),
      m_vbo(0),
      m_vboSize(0),
      m_radius(0.05f)
{
}

ScaleTool::~ScaleTool()
{
    deleteVBO();
}

void
ScaleTool::update()
{
    m_active = SelectionTool::hasScalableSelection( m_center );
}

void
ScaleTool::renderAxis( unsigned int i ) const
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
    glDrawArrays( GL_QUADS, 2, m_vboSize-2 );
    glDisableClientState( GL_VERTEX_ARRAY );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glPopMatrix();
}

void
ScaleTool::renderCenter() const
{
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glm::mat4 translate = glm::translate( glm::mat4(1.f), glm::vec3(m_center.x, m_center.y-(1.f-m_radius), m_center.z) );
    glMultMatrixf( glm::value_ptr(translate) );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, sizeof(vec3), (void*)(0) );
    glDrawArrays( GL_QUADS, 2, m_vboSize-2 );
    glDisableClientState( GL_VERTEX_ARRAY );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glPopMatrix();
}

void
ScaleTool::render()
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
        glColor3f( 1, 1, 0 );
        renderCenter();
        glPopAttrib();
        glPopAttrib();
        glPopAttrib();
    }
}

unsigned int
ScaleTool::getAxisPick() const
{
    unsigned int pick = Picker::NO_PICK;
    if ( m_active ) {
        m_panel->m_viewport->loadPickMatrices( UserInput::mousePos(), 1.f );
        Picker picker( 4 );
        for ( unsigned int i = 0; i < 3; ++i ) {
            picker.setObjectIndex( i );
            renderAxis( i );
        }
        picker.setObjectIndex( 3 );
        renderCenter();
        pick = picker.getPick();
        m_panel->m_viewport->popMatrices();
    }
    return pick;
}

void
ScaleTool::mousePressed()
{
    if ( m_active ) {
        m_transform = m_transformInverse = glm::mat4(1.f);
        m_axisSelection = getAxisPick();
        m_scaling = ( m_axisSelection != Picker::NO_PICK );
        if ( m_axisSelection == Picker::NO_PICK ) {
            SelectionTool::mousePressed();
        } else if ( m_axisSelection == 3 ) {
            m_mouseDownPos = UserInput::mousePos();
        }
    } else {
        SelectionTool::mousePressed();
    }
    update();
}

float
ScaleTool::intersectAxis( const glm::ivec2 &mouse ) const
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
    return vec3::dot( a, point-m_center );
}

void
ScaleTool::mouseMoved()
{
    if ( m_scaling ) {
        const glm::ivec2 &p0 = UserInput::mousePos() - UserInput::mouseMove();
        const glm::ivec2 &p1 = UserInput::mousePos();
        glm::mat4 transform = glm::mat4(1.f);
        if ( m_axisSelection < 3 ) {
            float t0 = intersectAxis( p0 );
            float t1 = intersectAxis( p1 );
            if ( fabsf(t1) > 1e-6 ) {
                float t = t1/t0;
                glm::vec3 scale = glm::vec3(1,1,1); scale[m_axisSelection] = t;
                transform = glm::scale( glm::mat4(1.f), scale );
            }
        } else {
            float d = 1.f + SCALE * ( p1.x - m_mouseDownPos.x );
            if ( fabsf(d) > 1e-6 ) {
                m_transform = glm::scale( glm::mat4(1.f), glm::vec3(d,d,d) );
                transform = m_transform * m_transformInverse;
                float *i = glm::value_ptr(m_transform);
                m_transformInverse = glm::mat4( 1.f/i[0], 0.f, 0.f, 0.f,
                                                0.f, 1.f/i[5], 0.f, 0.f,
                                                0.f, 0.f, 1.f/i[10], 0.f,
                                                0.f, 0.f, 0.f, 1.f );
            }
        }
        for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
            if ( (*it)->hasRenderable() && (*it)->getRenderable()->isSelected() &&
                 (*it)->getType() != SceneNode::SIMULATION_GRID ) {
                (*it)->applyTransformation( transform );
            }
        }
    }
}

void
ScaleTool::mouseReleased()
{
    m_axisSelection = Picker::NO_PICK;
    m_scaling = false;
    SelectionTool::mouseReleased();
    update();
}

bool
ScaleTool::hasVBO() const
{
    return m_vbo > 0 && glIsBuffer( m_vbo );
}

void
ScaleTool::buildVBO()
{
    deleteVBO();

    QVector<vec3> data;
    data += vec3( 0, 0, 0 );
    data += vec3( 0, 1, 0 );

    static const int resolution = 60;
    static const float dAngle = 2.f*M_PI/resolution;

    vec3 center = vec3( 0, 1-m_radius, 0 );

    for ( int i = 0; i < resolution; ++i ) {
        float theta0 = i*dAngle;
        float theta1 = (i+1)*dAngle;
        float y0 = cosf(theta0);
        float y1 = cosf(theta1);
        float r0 = sinf(theta0);
        float r1 = sinf(theta1);
        for ( int j = 0; j < resolution; ++j ) {
            float phi0 = j*dAngle;
            float phi1 = (j+1)*dAngle;
            float x0 = cosf(phi0);
            float x1 = cosf(phi1);
            float z0 = -sinf(phi0);
            float z1 = -sinf(phi1);
            data += ( center + m_radius*vec3(r0*x0, y0, r0*z0) );
            data += ( center + m_radius*vec3(r1*x0, y1, r1*z0) );
            data += ( center + m_radius*vec3(r1*x1, y1, r1*z1) );
            data += ( center + m_radius*vec3(r0*x1, y0, r0*z1) );
        }
    }

    glGenBuffers( 1, &m_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glBufferData( GL_ARRAY_BUFFER, data.size()*sizeof(vec3), data.data(), GL_STATIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    m_vboSize = data.size();
}

void
ScaleTool::deleteVBO()
{
    if ( m_vbo > 0 ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
        if ( glIsBuffer(m_vbo) ) glDeleteBuffers( 1, &m_vbo );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        m_vbo = 0;
    }
}
