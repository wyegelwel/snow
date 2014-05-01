/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   selectiontool.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 20 Apr 2014
**
**************************************************************************/

#include "selectiontool.h"

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/mat4x4.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "common/renderable.h"
#include "cuda/vector.h"
#include "geometry/bbox.h"
#include "scene/scene.h"
#include "scene/scenenode.h"
#include "scene/scenenodeiterator.h"
#include "ui/picker.h"
#include "ui/userinput.h"
#include "ui/viewpanel.h"
#include "viewport/viewport.h"

SelectionTool::SelectionTool( ViewPanel *panel ,Type t)
    : Tool(panel,t)
{
}

SelectionTool::~SelectionTool()
{
}

void
SelectionTool::mousePressed()
{
    Tool::mousePressed();
}

void
SelectionTool::mouseReleased()
{
    if ( m_mouseDown ) {
        SceneNode *selected = getSelectedSceneNode();
        if ( UserInput::shiftKey() ) {
            if ( selected ) selected->getRenderable()->setSelected( !selected->getRenderable()->isSelected() );
        } else {
            clearSelection();
            if ( selected ) selected->getRenderable()->setSelected( true );
        }
        Tool::mouseReleased();
    }
}

void
SelectionTool::clearSelection()
{
//    for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
//        if ( (*it)->hasRenderable() ) {
//            (*it)->getRenderable()->setSelected( false );
//        }
//    }
    m_panel->clearSelection();
}

SceneNode*
SelectionTool::getSelectedSceneNode()
{
    m_panel->m_viewport->loadPickMatrices( UserInput::mousePos(), 3.f );

    SceneNode *clicked = NULL;

    QList<SceneNode*> renderables;
    for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() ) {
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
            renderables[i]->getRenderable()->renderForPicker();
            glPopMatrix();
        }
        unsigned int index = picker.getPick();
        if ( index != Picker::NO_PICK ) {
            clicked = renderables[index];
        }
    }

    m_panel->m_viewport->popMatrices();
    return clicked;
}

bool
SelectionTool::hasSelection( vec3 &center ) const
{
    center = vec3( 0, 0, 0 );
    int count = 0;
    for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() && (*it)->getRenderable()->isSelected() ) {
            center += (*it)->getCentroid();
            count++;
        }
    }
    center /= (float)count;
    return ( count > 0 );
}

bool
SelectionTool::hasRotatableSelection( vec3 &center ) const
{
    center = vec3( 0, 0, 0 );
    int count = 0;
    for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() && (*it)->getRenderable()->isSelected() &&
             (*it)->getType() != SceneNode::SIMULATION_GRID ) {
            center += (*it)->getCentroid();
            count++;
        }
    }
    center /= (float)count;
    return ( count > 0 );
}

bool
SelectionTool::hasScalableSelection( vec3 &center ) const
{
    return hasRotatableSelection( center );
}
