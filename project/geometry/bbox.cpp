/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   bbox.cpp
**   Author: mliberma
**   Created: 10 Apr 2014
**
**************************************************************************/

#include "bbox.h"

#include <GL/gl.h>
#include <glm/gtc/type_ptr.hpp>

void
BBox::render()
{
    {
        glm::vec3 corners[8];
        glm::vec3 corner;
        for ( int x = 0, index = 0; x <= 1; x++ ) {
            corner.x = (x?m_max:m_min).x;
            for ( int y = 0; y <= 1; y++ ) {
                corner.y = (y?m_max:m_min).y;
                for ( int z = 0; z <= 1; z++, index++ ) {
                    corner.z = (z?m_max:m_min).z;
                    corners[index] = corner;
                }
            }
        }

        glBegin( GL_LINES );

        glVertex3fv( glm::value_ptr(corners[0]) );
        glVertex3fv( glm::value_ptr(corners[1]) );

        glVertex3fv( glm::value_ptr(corners[1]) );
        glVertex3fv( glm::value_ptr(corners[3]) );

        glVertex3fv( glm::value_ptr(corners[3]) );
        glVertex3fv( glm::value_ptr(corners[2]) );

        glVertex3fv( glm::value_ptr(corners[2]) );
        glVertex3fv( glm::value_ptr(corners[0]) );

        glVertex3fv( glm::value_ptr(corners[2]) );
        glVertex3fv( glm::value_ptr(corners[6]) );

        glVertex3fv( glm::value_ptr(corners[3]) );
        glVertex3fv( glm::value_ptr(corners[7]) );

        glVertex3fv( glm::value_ptr(corners[1]) );
        glVertex3fv( glm::value_ptr(corners[5]) );

        glVertex3fv( glm::value_ptr(corners[0]) );
        glVertex3fv( glm::value_ptr(corners[4]) );

        glVertex3fv( glm::value_ptr(corners[6]) );
        glVertex3fv( glm::value_ptr(corners[7]) );

        glVertex3fv( glm::value_ptr(corners[7]) );
        glVertex3fv( glm::value_ptr(corners[5]) );

        glVertex3fv( glm::value_ptr(corners[5]) );
        glVertex3fv( glm::value_ptr(corners[4]) );

        glVertex3fv( glm::value_ptr(corners[4]) );
        glVertex3fv( glm::value_ptr(corners[6]) );

        glEnd();

    }
}
