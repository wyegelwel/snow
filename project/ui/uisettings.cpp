/*!
   @file   uisettings.cpp
   @author Max Liberman (max_liberman@brown.edu)
   @date   2013
*/

#include <QSettings>

#include "common/common.h"
#include "ui/uisettings.h"

UiSettings* UiSettings::INSTANCE = NULL;

UiSettings*
UiSettings::instance()
{
    if ( !INSTANCE ) {
        INSTANCE = new UiSettings();
    }
    return INSTANCE;
}

void
UiSettings::deleteInstance()
{
    SAFE_DELETE( INSTANCE );
}

void
UiSettings::loadSettings()
{
    QSettings s( "CS224", "snow" );

    fillNumParticles() = s.value( "fillNumParticles", 512*128 ).toInt();
    fillResolution() = s.value( "fillResolution", 0.05f ).toFloat();
}

void
UiSettings::saveSettings()
{
    QSettings s( "CS224", "snow" );

    s.setValue( "fillNumParticles", fillNumParticles() );
    s.setValue( "fillResolution", fillResolution() );
}
