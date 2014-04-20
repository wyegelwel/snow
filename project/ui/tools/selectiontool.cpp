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

#include "common/renderable.h"
#include "cuda/vector.cu"
#include "geometry/bbox.h"
#include "scene/scene.h"
#include "scene/scenenode.h"
#include "scene/scenenodeiterator.h"
#include "ui/picker.h"
#include "ui/userinput.h"
#include "ui/viewpanel.h"
#include "viewport/viewport.h"

SelectionTool::SelectionTool( ViewPanel *panel )
    : Tool(panel)
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
    for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() ) {
            (*it)->getRenderable()->setSelected( false );
        }
    }
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
            picker.setObjectIndex(i);
            renderables[i]->getRenderable()->renderForPicker();
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
SelectionTool::hasSelection() const
{
    for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() && (*it)->getRenderable()->isSelected() ) {
            return true;
        }
    }
    return false;
}

vec3
SelectionTool::getSelectionCenter() const
{
    BBox box;
    for ( SceneNodeIterator it = m_panel->m_scene->begin(); it.isValid(); ++it ) {
        if ( (*it)->hasRenderable() && (*it)->getRenderable()->isSelected() ) {
            box += (*it)->getBBox();
        }
    }
    return box.center();
}
