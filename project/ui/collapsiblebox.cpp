/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   collapsiblebox.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 23 Apr 2014
**
**************************************************************************/

#include <QLayout>
#include <QMouseEvent>
#include <QPalette>

#include "collapsiblebox.h"

#include "common/common.h"

CollapsibleBox::CollapsibleBox( QWidget *widget )
    : QGroupBox(widget),
      m_clicked(false),
      m_collapsed(false)
{
    this->setAutoFillBackground( true );
}

void
CollapsibleBox::mousePressEvent( QMouseEvent *event )
{
    if ( !childrenRect().contains( event->pos() ) ) {
        QPalette palette = this->palette();
        m_color = palette.window().color();
        palette.setColor( QPalette::Window, QColor(155, 155, 155) );
        this->setPalette( palette );
        m_clicked = true;
    }
}

void
CollapsibleBox::mouseReleaseEvent( QMouseEvent *event )
{
    if ( m_clicked ) {
        for ( int i = 0; i < children().size(); ++i ) {
            QObject *child = this->children()[i];
            child->setProperty( "visible", m_collapsed );
        }
        m_collapsed = !m_collapsed;
        if ( m_collapsed ) {
            this->setMaximumHeight( this->fontMetrics().height() + this->fontMetrics().descent() );
        } else {
            this->setMaximumHeight( 16777215 );
        }
        QPalette palette = this->palette();
        palette.setColor( QPalette::Window, m_color );
        this->setPalette( palette );
        m_clicked = false;
        updateGeometry();
    }
}
