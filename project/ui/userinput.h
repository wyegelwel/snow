/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   userinput.h
**   Author: mliberma
**   Created: 7 Apr 2014
**
**************************************************************************/

#ifndef USERINPUT_H
#define USERINPUT_H

#include <QMouseEvent>
#include <glm/vec2.hpp>

class UserInput
{

public:

    static UserInput* instance();
    static void deleteInstance();

    static void update( QMouseEvent *event );

    static glm::ivec2 mousePos();
    static glm::ivec2 mouseMove();

    static bool leftMouse();
    static bool rightMouse();
    static bool middleMouse();

    static bool altKey();
    static bool ctrlKey();
    static bool shiftKey();

private:

    glm::ivec2 m_mousePos;
    glm::ivec2 m_mouseMove;
    Qt::MouseButton m_button;
    Qt::KeyboardModifiers m_modifiers;

    static UserInput *m_instance;

    UserInput();



};

#endif // USERINPUT_H
