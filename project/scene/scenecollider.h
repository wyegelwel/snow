/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   scenecollider.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 29 Apr 2014
**
**************************************************************************/

#ifndef SCENECOLLIDER_H
#define SCENECOLLIDER_H

#include "common/renderable.h"

struct BBox;
struct Mesh;
struct ImplicitCollider;

class SceneCollider : public Renderable {

public:

    SceneCollider( ImplicitCollider *collider );
    virtual ~SceneCollider();

    virtual void render();
    virtual void renderForPicker();
    virtual void renderVelForPicker();
    virtual void updateMeshVel();

//    virtual void setVelMag(const float m){m_mesh->setVelMag(m)};

    static constexpr float SphereRadius() { return .01f; }

    virtual BBox getBBox( const glm::mat4 &ctm );
    virtual vec3 getCentroid( const glm::mat4 &ctm );

    void initializeMesh();

    virtual void setSelected( bool selected );

    virtual void setCTM(const glm::mat4 &ctm);

    ImplicitCollider* getImplicitCollider() { return m_collider; }

private:

    ImplicitCollider *m_collider;
    Mesh *m_mesh;

};

#endif // SCENECOLLIDER_H
