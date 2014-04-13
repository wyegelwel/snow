/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   bbox.h
**   Author: mliberma
**   Created: 10 Apr 2014
**
**************************************************************************/

#ifndef BBOX_H
#define BBOX_H

#include <cmath>
#include <glm/common.hpp>
#include <glm/vec3.hpp>

#include "common/math.h"
#include "common/renderable.h"

#include "sim/grid.h"

class BBox : public Renderable
{

private:

    glm::vec3 m_min;
    glm::vec3 m_max;

public:

    inline BBox() { reset(); }
    inline BBox( const BBox &other ) : m_min(other.m_min), m_max(other.m_max) {}

    inline BBox( const glm::vec3 &p ) { m_min = m_max = p; }

    inline BBox( const glm::vec3 &p0, const glm::vec3 &p1 )
    {
        m_min = glm::min( p0, p1 );
        m_max = glm::max( p0, p1 );
    }

    inline void reset()
    {
        m_min = glm::vec3( INFINITY, INFINITY, INFINITY );
        m_max = glm::vec3( -INFINITY, -INFINITY, -INFINITY );
    }

    inline glm::vec3 center() const { return 0.5f*(m_max+m_min); }

    inline glm::vec3 min() const { return m_min; }
    inline glm::vec3 max() const { return m_max; }

    inline bool isEmpty() const { return m_min.x > m_max.x; }

    inline bool contains( const glm::vec3 &point ) const
    {
        return ( point.x >= m_min.x && point.x <= m_max.x ) &&
                ( point.y >= m_min.y && point.y <= m_max.y ) &&
                ( point.z >= m_min.z && point.z <= m_max.z );
    }

    inline glm::vec3 size() const { return m_max - m_min; }
    inline float width() const { return m_max.x - m_min.x; }
    inline float height() const { return m_max.y - m_min.y; }
    inline float depth() const { return m_max.z - m_max.z; }

    inline float volume() const { glm::vec3 s = m_max-m_min; return s.x*s.y*s.z; }
    inline float surfaceArea() const { glm::vec3 s = m_max-m_min; return 2*(s.x*s.y+s.x*s.z+s.y*s.z); }

    inline void fix( float h )
    {
        glm::vec3 c = 0.5f*(m_min+m_max);
        glm::vec3 d = h*glm::ceil((m_max-m_min)/h)/2.f;
        m_min = c - d;
        m_max = c + d;
    }

    inline Grid toGrid( float h ) const
    {
        BBox box(*this);
        box.expandAbs( h );
        box.fix( h );
        Grid grid;
        glm::vec3 dimf = glm::round( (box.max()-box.min())/h );
        grid.dim = glm::ivec3( dimf.x, dimf.y, dimf.z );
        grid.h = h;
        grid.pos = box.min();
        return grid;
    }

    // Expand box by absolute distances
    inline void expandAbs( float d ) { m_min -= glm::vec3(d, d, d); m_max += glm::vec3(d, d, d); }
    inline void expandAbs( const glm::vec3 &d ) { m_min -= d; m_max += d; }

    // Expand box relative to current size
    inline void expandRel( float d ) { glm::vec3 dd = d*(m_max-m_min); m_min -= dd; m_max += dd; }
    inline void expandRel( const glm::vec3 &d ) { glm::vec3 dd = d*(m_max-m_min); m_min -= dd; m_max += dd; }

    // Merge two bounding boxes
    inline BBox& operator += ( const BBox &rhs )
    {
        m_min = glm::min( m_min, rhs.m_min );
        m_max = glm::max( m_max, rhs.m_max );
        return *this;
    }

    inline BBox  operator +  ( const BBox &rhs ) const
    {
        return BBox( glm::min(m_min,rhs.m_min), glm::max(m_max,rhs.m_max) );
    }

    // Incorporate point into bounding box
    inline BBox& operator += ( const glm::vec3 &rhs )
    {
        m_min = glm::min( m_min, rhs );
        m_max = glm::max( m_max, rhs );
        return *this;
    }

    inline BBox  operator +  ( const glm::vec3 &rhs ) const
    {
        return BBox( glm::min(m_min,rhs), glm::max(m_max,rhs) );
    }

    void render();

};

#endif // BBOX_H
