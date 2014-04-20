/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   databinding.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 17 Apr 2014
**
**************************************************************************/

#ifndef UISETTINGS_H
#define UISETTINGS_H

#define DEFINE_SETTING( TYPE, NAME )                            \
    private:                                                    \
        TYPE m_##NAME;                                          \
    public:                                                     \
        static TYPE& NAME() { return instance()->m_##NAME; }    \


#include <QPoint>
#include <QSize>

class UiSettings
{

public:

    static UiSettings* instance();
    static void deleteInstance();

    static void loadSettings();
    static void saveSettings();

protected:

    UiSettings() {}
    virtual ~UiSettings() {}

private:

    static UiSettings *INSTANCE;

    DEFINE_SETTING( QPoint, windowPosition )
    DEFINE_SETTING( QSize, windowSize )

    DEFINE_SETTING( int, fillNumParticles )
    DEFINE_SETTING( float, fillResolution )

    DEFINE_SETTING( bool, exportSimulation )

};

#endif // UISETTINGS_H