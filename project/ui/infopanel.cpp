/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   infopanel.cpp
**   Author: mliberma
**   Created: 10 Apr 2014
**
**************************************************************************/

#include <QFontMetrics>
#include <QGridLayout>
#include <QLabel>

#include "infopanel.h"

#include "common/common.h"
#include "ui/viewpanel.h"

InfoPanel::InfoPanel( ViewPanel *panel )
    : m_panel(panel)
{
    m_font = QFont( "Helvetica", 10 );
    m_spacing = 2;
    m_margin = 6;
}

void
InfoPanel::addInfo( const QString &key,
                    const QString &value )
{
    m_info.insert( key, Entry(key, value) );
    m_order += key;
    updateLayout();
}

void
InfoPanel::setInfo( const QString &key,
                    const QString &value,
                    bool layout )
{
    if ( !m_info.contains(key) ) {
        m_info.insert( key, Entry(key, value) );
        m_order += key;
    } else {
        m_info[key].value = value;
    }
    if ( layout ) updateLayout();
}

void
InfoPanel::removeInfo( const QString &key )
{
    if ( m_info.contains(key) ) {
        m_info.remove( key );
        m_order.removeAll( key );
    }
    updateLayout();
}

void
InfoPanel::updateLayout()
{
    QFontMetrics metrics( m_font );
    int h = metrics.height();
    int y = metrics.ascent() + m_margin;
    int x0 = m_margin;
    for ( int i = 0; i < m_order.size(); ++i ) {
        Entry &entry = m_info[m_order[i]];
        entry.pos.y = y;
        entry.pos.x = x0;
        y += (h+m_spacing);
    }
}

void
InfoPanel::render()
{
    glPushAttrib( GL_COLOR_BUFFER_BIT );
    glColor4f( 1.f, 1.f, 1.f, 1.f );
    for ( QHash<QString, Entry>::const_iterator it = m_info.begin(); it != m_info.end(); ++it ) {
        const Entry &entry = it.value();
        m_panel->renderText( entry.pos.x, entry.pos.y, QString("%1: %2").arg(entry.key, entry.value) );
    }
    glPopAttrib();
}
