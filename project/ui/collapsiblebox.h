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
    Q_PROPERTY( bool collapsed READ isCollapsed WRITE setCollapsed )

public:

    explicit CollapsibleBox( QWidget *parent );
    ~CollapsibleBox() {}

    bool isCollapsed() const { return m_collapsed; }

public slots:

    virtual void mousePressEvent( QMouseEvent *event );
    virtual void mouseReleaseEvent( QMouseEvent* );

    virtual void setTitle( const QString &title );

    void setCollapsed( bool collapsed );

protected:

    QString m_rawTitle;

    bool m_clicked;
    bool m_collapsed;

    static void setWidgetPalette( QWidget *widget , const QPalette &palette );

};

#endif // COLLAPSIBLEBOX_H
