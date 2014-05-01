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

QVariant
UiSettings::getSetting( const QString &name, const QVariant &d )
{
    QSettings s( "CS224", "snow" );
    return s.value( name, d );
}

void
UiSettings::setSetting( const QString &name, const QVariant &value )
{
    QSettings s( "CS224", "snow" );
    s.setValue( name, value );
}

void
UiSettings::loadSettings()
{
    QSettings s( "CS224", "snow" );

    windowPosition() = s.value( "windowPosition", QPoint(0,0) ).toPoint();
    windowSize() = s.value( "windowSize", QSize(1000,800) ).toSize();

    fillNumParticles() = s.value( "fillNumParticles", 512*128 ).toInt();
    fillResolution() = s.value( "fillResolution", 0.05f ).toFloat();
    fillDensity() = s.value( "fillDensity", 150.f ).toFloat();

    exportDensity() = s.value("exportDensity", false).toBool();
    exportVelocity() = s.value("exportVelocity", false).toBool();

    exportFPS() = s.value("exportFPS", 24).toInt();
    maxTime() = s.value("maxTime", 3).toFloat();

    gridPosition() = vec3( s.value("gridPositionX", 0.f).toFloat(),
                           s.value("gridPositionY", 0.f).toFloat(),
                           s.value("gridPositionZ", 0.f).toFloat() );


    gridDimensions() = glm::ivec3( s.value("gridDimensionX", 128).toInt(),
                                   s.value("gridDimensionY", 128).toInt(),
                                   s.value("gridDimensionZ", 128).toInt() );

    gridResolution() = s.value( "gridResolution", 0.05f ).toFloat();

    timeStep() = s.value( "timeStep", 1e-5 ).toFloat();
    implicit() = s.value( "implicit", true ).toBool();
    materialPreset() = s.value( "materialPreset", MAT_DEFAULT).toInt();

    showContainers() = s.value( "showContainers", true ).toBool();
    showContainersMode() = s.value( "showContainersMode", WIREFRAME ).toInt();
    showColliders() = s.value( "showColliders", true ).toBool();
    showCollidersMode() = s.value( "showCollidersMode", SOLID ).toInt();
    showGrid() = s.value( "showGrid", false ).toBool();
    showGridMode() = s.value( "showGridMode", MIN_FACE_CELLS ).toInt();
    showGridData() = s.value( "showGridData", false ).toBool();
    showGridDataMode() = s.value( "showGridDataMode", NODE_DENSITY ).toInt();
    showParticles() = s.value( "showParticles", true ).toBool();
    showParticlesMode() = s.value( "showParticlesMode", PARTICLE_MASS ).toInt();

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
    s.setValue( "fillDensity", fillDensity() );

    s.setValue("exportDensity", exportDensity());
    s.setValue("exportVelocity",exportVelocity());
    s.setValue( "exportFPS", exportFPS());
    s.setValue( "maxTime", maxTime());

    s.setValue( "gridPositionX", gridPosition().x );
    s.setValue( "gridPositionY", gridPosition().y );
    s.setValue( "gridPositionZ", gridPosition().z );

    s.setValue( "gridDimensionX", gridDimensions().x );
    s.setValue( "gridDimensionY", gridDimensions().y );
    s.setValue( "gridDimensionZ", gridDimensions().z );

    s.setValue( "gridResolution", gridResolution() );

    s.setValue( "timeStep", timeStep() );
    s.setValue( "implicit", implicit() );
    s.setValue("materialPreset", materialPreset());

    s.setValue( "showContainers", showContainers() );
    s.setValue( "showContainersMode", showContainersMode() );
    s.setValue( "showColliders", showColliders() );
    s.setValue( "showCollidersMode", showCollidersMode() );
    s.setValue( "showGrid", showGrid() );
    s.setValue( "showGridMode", showGridMode() );
    s.setValue( "showGridData", showGridData() );
    s.setValue( "showGridDataMode", showGridDataMode() );
    s.setValue( "showParticles", showParticles() );
    s.setValue( "showParticlesMode", showParticlesMode() );
}

Grid
UiSettings::buildGrid( const glm::mat4 &ctm )
{
    Grid grid;
    vec3 p = UiSettings::gridPosition();
    glm::vec4 point = ctm * glm::vec4(p.x,p.y,p.z,1);
    grid.pos = vec3( point.x, point.y, point.z );
    grid.dim = UiSettings::gridDimensions();
    grid.h = UiSettings::gridResolution();
    return grid;
}
