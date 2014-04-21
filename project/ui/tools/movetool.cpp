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

#include "movetool.h"

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"

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
      m_center(0,0,0)
{
}

MoveTool::~MoveTool()
{
}

void
MoveTool::update()
{
    if ( (m_active = SelectionTool::hasSelection()) ) {
        m_center = SelectionTool::getSelectionCenter();
    }
}

void
MoveTool::renderAxis( const vec3 &axis ) const
{
    glLineWidth( 2.f );
    glBegin( GL_LINES );
    glVertex3fv( m_center.data );
    glVertex3fv( (m_center+axis).data );
    glEnd();
}

void
MoveTool::render()
{
    if ( m_active ) {
        glPushAttrib( GL_DEPTH_BUFFER_BIT );
        glDisable( GL_DEPTH_TEST );
        glPushAttrib( GL_COLOR_BUFFER_BIT );
        for ( unsigned int i = 0; i < 3; ++i ) {
            vec3 color( 0, 0, 0 );
            if ( m_axisSelection == i ) color = vec3( 1, 1, 0 );
            else color[i] = 1;
            glColor3fv( color.data );
            vec3 axis = vec3(0,0,0); axis[i] = 1;
            renderAxis( axis );
        }
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
        for ( int i = 0; i < 3; ++i ) {
            picker.setObjectIndex( i );
            vec3 axis = vec3(0,0,0); axis[i] = 1;
            renderAxis( axis );
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
//                if ( (*it)->getType() == SceneNode::IMPLICIT_COLLIDER)  {
//                    dynamic_cast<Collider*>((*it)->getRenderable())->getImplicitCollider()->center += translate;
//                }
//                else  {
//                    (*it)->applyTransformation( transform );
//                }
//                if ( (*it)->getType() == SceneNode::SIMULATION_GRID ) {
//                    UiSettings::gridPosition() += translate;
//                }
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
