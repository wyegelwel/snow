/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   infopanel.h
**   Author: mliberma
**   Created: 10 Apr 2014
**
**************************************************************************/

#ifndef INFOPANEL_H
#define INFOPANEL_H

#include <QFont>
#include <QHash>
#include <QString>

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include <glm/vec2.hpp>

#include "common/renderable.h"

/*
 * An informational GUI element that manages the layout
 * of a list of key-value pairs to be displayed.
 */

class ViewPanel;

class InfoPanel
{

    struct Entry {
        QString key;
        QString value;
        glm::ivec2 pos;
        Entry() {}
        Entry( const QString &k, const QString &v ) : key(k), value(v) {}
        Entry( const Entry &entry ) : key(entry.key), value(entry.value), pos(entry.pos) {}
    };

public:

    InfoPanel( ViewPanel *panel );
    virtual ~InfoPanel() {}

    void setFont( const QFont &font ) { m_font = font; }

    void addInfo( const QString &key, const QString &value = QString() );
    void setInfo( const QString &key, const QString &value, bool layout = true );
    void removeInfo( const QString &key );

    void render();

private:

    ViewPanel *m_panel;
    QHash<QString, Entry> m_info;
    QList<QString> m_order;

    QFont m_font;
    int m_spacing, m_margin;

    void updateLayout();

};

#endif // INFOPANEL_H
