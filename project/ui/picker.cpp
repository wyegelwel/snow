/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   picker.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 19 Apr 2014
**
**************************************************************************/

#include <GL/gl.h>

#include "ui/picker.h"

#include "common/common.h"

const unsigned int Picker::NO_PICK = INT_MAX;

Picker::Picker( int n )
    : m_picks(NULL)
{
    m_nObjects = n;
    if ( m_nObjects > 0 ) {
        m_picks = new PickRecord[m_nObjects];
        glSelectBuffer( 4*m_nObjects, (unsigned int *)m_picks );
        glRenderMode( GL_SELECT );
        glInitNames();
        glPushName(0);
        m_selectMode = true;
    }
}

Picker::~Picker()
{
    SAFE_DELETE_ARRAY( m_picks );
}

void
Picker::setObjectIndex( unsigned int i ) const
{
    glLoadName( i );
}

unsigned int
Picker::getPick()
{
    unsigned int index = NO_PICK;
    if ( (m_nObjects > 0) && m_selectMode ) {
        int hits = glRenderMode( GL_RENDER );
        unsigned int depth = ~0;
        for ( int i = 0; i < hits; i++ ) {
            const PickRecord &pick = m_picks[i];
            if ( pick.minDepth < depth ) {
                index = pick.name;
                depth = pick.minDepth;
            }
        }
        m_selectMode = false;
    }
    return index;
}

QList<unsigned int>
Picker::getPicks()
{
    QList<unsigned int> picks;
    if ( (m_nObjects > 0) && m_selectMode ) {
        int hits = glRenderMode( GL_RENDER );
        for ( int i = 0; i < hits; i++ ) {
            const PickRecord &pick = m_picks[i];
            picks += pick.name;
        }
        m_selectMode = false;
    }
    return picks;
}



