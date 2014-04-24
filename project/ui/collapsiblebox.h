/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   collapsiblebox.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 23 Apr 2014
**
**************************************************************************/

#ifndef COLLAPSIBLEBOX_H
#define COLLAPSIBLEBOX_H

#include <QGroupBox>

class CollapsibleBox : public QGroupBox
{

    Q_OBJECT

public:

    explicit CollapsibleBox( QWidget *parent );
    ~CollapsibleBox() {}

public slots:

    virtual void mousePressEvent( QMouseEvent *event );
    virtual void mouseReleaseEvent( QMouseEvent* );

protected:

    bool m_clicked;
    bool m_collapsed;

    static void setWidgetPalette( QWidget *widget , const QPalette &palette );

};

#endif // COLLAPSIBLEBOX_H
