/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   userinput.cpp
**   Author: mliberma
**   Created: 7 Apr 2014
**
**************************************************************************/

#include "userinput.h"

#include "common/common.h"

UserInput* UserInput::m_instance = NULL;

UserInput::UserInput()
{
    m_mousePos = glm::ivec2(0,0);
    m_mouseMove = glm::ivec2(0,0);
    m_button = Qt::NoButton;
    m_modifiers = Qt::NoModifier;
}

UserInput*
UserInput::instance()
{
    if ( m_instance == NULL ) {
        m_instance = new UserInput;
    }
    return m_instance;
}

void
UserInput::deleteInstance()
{
    SAFE_DELETE( m_instance );
}

void
UserInput::update( QMouseEvent *event )
{
    instance()->m_mouseMove = glm::ivec2( event->pos().x()-instance()->m_mousePos.x,
                                          event->pos().y()-instance()->m_mousePos.y );
    instance()->m_mousePos = glm::ivec2( event->pos().x(), event->pos().y() );
    instance()->m_button = event->button();
    instance()->m_modifiers = event->modifiers();
}

glm::ivec2
UserInput::mousePos()
{
    return instance()->m_mousePos;
}

glm::ivec2
UserInput::mouseMove()
{
    return instance()->m_mouseMove;
}

bool
UserInput::leftMouse()
{
    return instance()->m_button == Qt::LeftButton;
}

bool
UserInput::rightMouse()
{
    return instance()->m_button == Qt::RightButton;
}

bool
UserInput::middleMouse()
{
    return instance()->m_button == Qt::MiddleButton;
}

bool
UserInput::altKey()
{
    return (instance()->m_modifiers & Qt::AltModifier);
}

bool
UserInput::ctrlKey()
{
    return (instance()->m_modifiers & Qt::ControlModifier);
}

bool
UserInput::shiftKey()
{
    return (instance()->m_modifiers & Qt::ShiftModifier);
}
