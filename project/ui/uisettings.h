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
#include <QVariant>

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"

#include "cuda/vector.h"

struct Grid;

class UiSettings
{

public:

    enum MeshMode
    {
        WIREFRAME,
        SOLID,
        SOLID_AND_WIREFRAME
    };

    enum GridMode
    {
        BOX,
        MIN_FACE_CELLS,
        ALL_FACE_CELLS
    };

    enum GridDataMode
    {
        NODE_DENSITY,
        NODE_VELOCITY,
        NODE_SPEED,
        NODE_FORCE
    };

    enum ParticlesMode
    {
        PARTICLE_MASS,
        PARTICLE_VELOCITY,
        PARTICLE_SPEED,
        PARTICLE_SHADED,
        PARTICLE_STIFFNESS
    };

    enum SnowMaterialPreset
    {
        MAT_DEFAULT,
        MAT_CHUNKY
    };

public:

    static UiSettings* instance();
    static void deleteInstance();

    static void loadSettings();
    static void saveSettings();

    static QVariant getSetting( const QString &name, const QVariant &d = QVariant() );
    static void setSetting( const QString &name, const QVariant &value );

    static Grid buildGrid( const glm::mat4 &ctm );

protected:

    UiSettings() {}
    virtual ~UiSettings() {}

private:

    static UiSettings *INSTANCE;

    DEFINE_SETTING( QPoint, windowPosition )
    DEFINE_SETTING( QSize, windowSize )

    // filling
    DEFINE_SETTING( int, fillNumParticles )
    DEFINE_SETTING( float, fillDensity )
    DEFINE_SETTING( float, fillResolution )

    // exporting
    DEFINE_SETTING( bool, exportDensity )
    DEFINE_SETTING( bool, exportVelocity )
    DEFINE_SETTING( int, exportFPS)
    DEFINE_SETTING( float, maxTime)

    DEFINE_SETTING( vec3, gridPosition )
    DEFINE_SETTING( glm::ivec3, gridDimensions )
    DEFINE_SETTING( float, gridResolution )

    DEFINE_SETTING( float, timeStep )
    DEFINE_SETTING( int, materialPreset )

    DEFINE_SETTING( bool, showContainers )
    DEFINE_SETTING( int, showContainersMode )
    DEFINE_SETTING( bool, showColliders )
    DEFINE_SETTING( int, showCollidersMode )
    DEFINE_SETTING( bool, showGrid )
    DEFINE_SETTING( int, showGridMode )
    DEFINE_SETTING( bool, showGridData )
    DEFINE_SETTING( int, showGridDataMode )
    DEFINE_SETTING( bool, showParticles )
    DEFINE_SETTING( int, showParticlesMode )

    DEFINE_SETTING( glm::vec4, selectionColor )

};

#endif // UISETTINGS_H
