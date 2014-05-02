/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   VelocityTool.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 21 Apr 2014
**
**************************************************************************/

#include <GL/gl.h>

#include <QVector>

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "velocitytool.h"
#include "scene/scene.h"
#include "scene/scenenode.h"
#include "scene/scenenodeiterator.h"
#include "ui/picker.h"
#include "ui/userinput.h"
#include "ui/uisettings.h"
#include "ui/viewpanel.h"
#include "viewport/camera.h"
#include "viewport/viewport.h"

#include <iostream>

VelocityTool::VelocityTool( ViewPanel *panel,Type t )
    : SelectionTool(panel,t),
      m_axisSelection(Picker::NO_PICK),
      m_vecSelection(Picker::NO_PICK),
      m_active(false),
      m_rotating(false),
      m_scaling(false),
      m_center(0,0,0),
      m_scale(1.f),
      m_vbo(0)
{
}

VelocityTool::~VelocityTool()
{
    deleteVBO();
}

void
VelocityTool::update()
{
    if ( (m_active = SelectionTool::hasRotatableSelection(m_center)) ) {
        m_scale = Tool::getHandleSize( m_center );
    }
}

void
VelocityTool::renderAxis( unsigned int i ) const
{
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glm::mat4 translate = glm::translate( glm::mat4(1.f), glm::vec3(m_center) );
    glm::mat4 basis = glm::scale( Tool::getAxialBasis(i), glm::vec3(m_scale) );
    glMultMatrixf( glm::value_ptr(translate*basis) );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, sizeof(vec3), (void*)(0) );
    glLineWidth( 2.f );
    glDrawArrays( GL_LINE_LOOP, 0, m_vboSize );
    glDisableClientState( GL_VERTEX_ARRAY );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glPopMatrix();
}

void
VelocityTool::render()
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
            glColor3fv( Tool::getAxialColor((i==m_axisSelection)?3:i).data );
            renderAxis( i );
        }
        glPopAttrib();
        glPopAttrib();
        glPopAttrib();
    }
}



unsigned int
VelocityTool::getAxisPick() const
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

unsigned int
VelocityTool::getVelVecPick() const
{
    unsigned int pick = Picker::NO_PICK;

    m_panel->m_viewport->loadPickMatrices( UserInput::mousePos(), 3.f );

    SceneNode *clicked = NULL;

    QList<SceneNode*> renderables;
    for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() && (*it)->getRenderable()->isSelected()) {
            renderables += (*it);
        }
    }
    if ( !renderables.empty() ) {
        Picker picker( renderables.size() );
        for ( int i = 0; i < renderables.size(); ++i ) {
            glMatrixMode( GL_MODELVIEW );
            glPushMatrix();
            glMultMatrixf( glm::value_ptr(renderables[i]->getCTM()) );
            picker.setObjectIndex(i);
            renderables[i]->getRenderable()->renderVelForPicker();
            glPopMatrix();
        }
        pick = picker.getPick();
        if ( pick != Picker::NO_PICK ) {
            clicked = renderables[pick];
        }
    }

    return pick;
}

void
VelocityTool::mousePressed()
{
    if ( m_active ) {
        m_axisSelection = getAxisPick();
        m_vecSelection = getVelVecPick();
        m_rotating = ( m_axisSelection != Picker::NO_PICK && m_vecSelection == Picker::NO_PICK);
        m_scaling = (m_vecSelection != Picker::NO_PICK);
        if ( m_axisSelection == Picker::NO_PICK && m_vecSelection == Picker::NO_PICK) {
            SelectionTool::mousePressed();
        }
    } else {
        SelectionTool::mousePressed();
    }
    update();
}

float
VelocityTool::intersectVelVec( const glm::ivec2 &mouse,const glm::vec3 &velVec ) const
{
    glm::vec2 uv = glm::vec2( (float)mouse.x/m_panel->width(), (float)mouse.y/m_panel->height() );
    vec3 direction = m_panel->m_viewport->getCamera()->getCameraRay( uv );
    vec3 origin = m_panel->m_viewport->getCamera()->getPosition();
    unsigned int majorAxis = direction.majorAxis();
    int axis = majorAxis;
    if ( majorAxis == m_axisSelection ) {
        axis = ( majorAxis == 0 ) ? 1 : 0;
    }
    float t = (0-origin[axis])/direction[axis];
    vec3 point = origin + t*direction;
    return vec3::dot( velVec, point-m_center );
}

float
VelocityTool::intersectAxis( const glm::ivec2 &mouse ) const
{
    glm::vec2 uv = glm::vec2( (float)mouse.x/m_panel->width(), (float)mouse.y/m_panel->height() );
    vec3 direction = m_panel->m_viewport->getCamera()->getCameraRay( uv );
    vec3 origin = m_panel->m_viewport->getCamera()->getPosition();
    vec3 normal(0,0,0); normal[m_axisSelection] = 1.f;
    float t = (m_center[m_axisSelection]-origin[m_axisSelection])/direction[m_axisSelection];
    vec3 circle = (origin + t*direction)-m_center;
    float y = circle[(m_axisSelection+2)%3];
    float x = circle[(m_axisSelection+1)%3];
    return atan2( y, x );
}

void
VelocityTool::mouseMoved()
{
    if ( m_rotating ) {
        float theta0 = intersectAxis( UserInput::mousePos()-UserInput::mouseMove() );
        float theta1 = intersectAxis( UserInput::mousePos() );
        float theta = theta1-theta0;
        glm::mat4 Tinv = glm::translate( glm::mat4(1.f), glm::vec3(-m_center) );
        glm::mat4 T = glm::translate( glm::mat4(1.f), glm::vec3(m_center) );
        glm::vec3 axis(0,0,0); axis[m_axisSelection] = 1.f;
        glm::mat4 R = glm::rotate( glm::mat4(1.f), theta, axis );
        glm::mat4 transform = T * R * Tinv;
        for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
            if ( (*it)->hasRenderable() && (*it)->getRenderable()->isSelected() &&
                 (*it)->getType() != SceneNode::SIMULATION_GRID ) {
                (*it)->getRenderable()->rotateVelVec( transform );
                (*it)->getRenderable()->updateMeshVel();
                m_panel->checkSelected();
            }
//            else if((*it)->getType() == SceneNode::IMPLICIT_COLLIDER && (*it)->hasRenderable() && (*it)->getRenderable()->isSelected())  {
//                switch(dynamic_cast<SceneCollider*>((*it)->getRenderable())->getImplicitCollider()->type) {
//                    case SPHERE:
//                        break;
//                    case HALF_PLANE:
//                        (*it)->applyTransformation( transform );
//                        break;
//                    default:
//                        break;
//                }
//            }
//            else {}
        }
    }
    if( m_scaling)  {
        const float scale_factor=50.0f;
        const glm::ivec2 &p0 = UserInput::mousePos() - UserInput::mouseMove();
        const glm::ivec2 &p1 = UserInput::mousePos();
        for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
            if ( (*it)->hasRenderable() && (*it)->getRenderable()->isSelected() &&
                 (*it)->getType() != SceneNode::SIMULATION_GRID ) {
                float t0,t1;
                glm::vec3 velVec = (*it)->getRenderable()->getVelVec();
                t0 = intersectVelVec(p0,velVec);
                t1 = intersectVelVec(p1,velVec);
                (*it)->getRenderable()->setVelMag((*it)->getRenderable()->getVelMag() + (t1-t0)*scale_factor);
                std::cout << (*it)->getRenderable()->getVelMag() << std::endl;
                (*it)->getRenderable()->updateMeshVel();
//                emit m_panel->changeVelMag((*it)->getRenderable()->getVelMag());
                m_panel->checkSelected();
            }
         }
    }
    update();
}

void
VelocityTool::mouseReleased()
{
    m_axisSelection = Picker::NO_PICK;
    m_vecSelection = Picker::NO_PICK;
    m_scaling = false;
    m_rotating = false;
    SelectionTool::mouseReleased();
    update();
}

bool
VelocityTool::hasVBO() const
{
    return m_vbo > 0 && glIsBuffer( m_vbo );
}

void
VelocityTool::buildVBO()
{
    deleteVBO();

    QVector<vec3> data;

    static const int resolution = 60;
    static const float dTheta = 2.f*M_PI/resolution;
    for ( int i = 0; i < resolution; ++i )
        data += vec3( cosf(i*dTheta), 0.f, -sinf(i*dTheta) );

    glGenBuffers( 1, &m_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
    glBufferData( GL_ARRAY_BUFFER, data.size()*sizeof(vec3), data.data(), GL_STATIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    m_vboSize = data.size();

}

void
VelocityTool::deleteVBO()
{
    if ( m_vbo > 0 ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
        if ( glIsBuffer(m_vbo) ) glDeleteBuffers( 1, &m_vbo );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        m_vbo = 0;
    }
}
