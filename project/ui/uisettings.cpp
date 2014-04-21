/*!
   @file   uisettings.cpp
   @author Max Liberman (max_liberman@brown.edu)
   @date   2013
*/

#include <QSettings>

#include "common/common.h"
#include "geometry/grid.h"
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

    windowPosition() = s.value( "windowPosition", QPoint(0,0) ).toPoint();
    windowSize() = s.value( "windowSize", QSize(1000,800) ).toSize();

    fillNumParticles() = s.value( "fillNumParticles", 512*128 ).toInt();
    fillResolution() = s.value( "fillResolution", 0.05f ).toFloat();

    exportSimulation() = s.value( "exportSimulation", false ).toBool();

    gridPosition() = vec3( s.value("gridPositionX", 0.f).toFloat(),
                           s.value("gridPositionY", 0.f).toFloat(),
                           s.value("gridPositionZ", 0.f).toFloat() );

    gridDimensions() = glm::ivec3( s.value("gridDimensionX", 128).toInt(),
                                   s.value("gridDimensionY", 128).toInt(),
                                   s.value("gridDimensionZ", 128).toInt() );

    gridResolution() = s.value( "gridResolution", 0.05f ).toFloat();

    showMesh() = s.value( "showMesh", true ).toBool();
    showMeshMode() = s.value( "showMeshMode", WIREFRAME ).toInt();
    showGrid() = s.value( "showGrid", false ).toBool();
    showGridMode() = s.value( "showGridMode", HALF_CELLS ).toInt();
    showGridData() = s.value( "showGridData", false ).toBool();
    showGridDataMode() = s.value( "showGridDataMode", MASS ).toInt();
    showParticles() = s.value( "showParticles", true ).toBool();

    selectionColor() = glm::vec4( 0.302f, 0.773f, 0.839f, 1.f );
}

void
UiSettings::saveSettings()
{
    QSettings s( "CS224", "snow" );

    s.setValue( "windowPosition", windowPosition() );
    s.setValue( "windowSize", windowSize() );

    s.setValue( "fillNumParticles", fillNumParticles() );
    s.setValue( "fillResolution", fillResolution() );

    s.setValue( "exportSimulation", exportSimulation() );

    s.setValue( "gridPositionX", gridPosition().x );
    s.setValue( "gridPositionY", gridPosition().y );
    s.setValue( "gridPositionZ", gridPosition().z );

    s.setValue( "gridDimensionX", gridDimensions().x );
    s.setValue( "gridDimensionY", gridDimensions().y );
    s.setValue( "gridDimensionZ", gridDimensions().z );

    s.setValue( "gridResolution", gridResolution() );

    s.setValue( "showMesh", showMesh() );
    s.setValue( "showMeshMode", showMeshMode() );
    s.setValue( "showGrid", showGrid() );
    s.setValue( "showGridMode", showGridMode() );
    s.setValue( "showGridData", showGridData() );
    s.setValue( "showGridDataMode", showGridDataMode() );
    s.setValue( "showParticles", showParticles() );
}

Grid
UiSettings::buildGrid()
{
    Grid grid;
    grid.pos = UiSettings::gridPosition();
    grid.dim = UiSettings::gridDimensions();
    grid.h = UiSettings::gridResolution();
    return grid;
}
