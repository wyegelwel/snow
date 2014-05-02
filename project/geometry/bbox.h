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

#include "common/math.h"
#include "common/renderable.h"

#include "geometry/grid.h"
#include "cuda/vector.h"

/**
 * @brief Axis-aligned bounding box
 */
class BBox : public Renderable
{

private:

    vec3 m_min;
    vec3 m_max;

public:

    inline BBox() { reset(); }
    inline BBox( const BBox &other ) : m_min(other.m_min), m_max(other.m_max) {}

    inline BBox( const Grid &grid )
    {
        m_min = grid.pos;
        m_max = grid.pos + grid.h * vec3( grid.dim.x, grid.dim.y, grid.dim.z );
    }

    inline BBox( const vec3 &p ) { m_min = m_max = p; }

    inline BBox( const vec3 &p0, const vec3 &p1 )
    {
        m_min = vec3::min( p0, p1 );
        m_max = vec3::max( p0, p1 );
    }

    inline void reset()
    {
        m_min = vec3( INFINITY, INFINITY, INFINITY );
        m_max = vec3( -INFINITY, -INFINITY, -INFINITY );
    }

    inline vec3 center() const { return 0.5f*(m_max+m_min); }

    inline vec3 min() const { return m_min; }
    inline vec3 max() const { return m_max; }

    inline bool isEmpty() const { return m_min.x > m_max.x; }

    inline bool contains( const vec3 &point ) const
    {
        return ( point.x >= m_min.x && point.x <= m_max.x ) &&
                ( point.y >= m_min.y && point.y <= m_max.y ) &&
                ( point.z >= m_min.z && point.z <= m_max.z );
    }

    inline vec3 size() const { return m_max - m_min; }
    inline float width() const { return m_max.x - m_min.x; }
    inline float height() const { return m_max.y - m_min.y; }
    inline float depth() const { return m_max.z - m_max.z; }

    inline int longestDim() const
    {
        vec3 size = m_max - m_min;
        return ( size.x > size.y ) ? ( (size.x > size.z) ? 0 : 2 ) : ( (size.y > size.z) ? 1 : 2 );
    }

    inline float longestDimSize() const
    {
        vec3 size = m_max - m_min;
        return ( size.x > size.y ) ? ( (size.x > size.z) ? size.x : size.z ) : ( (size.y > size.z) ? size.y : size.z );
    }

    inline float volume() const { vec3 s = m_max-m_min; return s.x*s.y*s.z; }
    inline float surfaceArea() const { vec3 s = m_max-m_min; return 2*(s.x*s.y+s.x*s.z+s.y*s.z); }

    inline void fix( float h )
    {
        vec3 c = 0.5f*(m_min+m_max);
        vec3 d = h*vec3::ceil((m_max-m_min)/h)/2.f;
        m_min = c - d;
        m_max = c + d;
    }

    inline Grid toGrid( float h ) const
    {
        BBox box(*this);
        box.expandAbs( h );
        box.fix( h );
        Grid grid;
        vec3 dimf = vec3::round( (box.max()-box.min())/h );
        grid.dim = glm::ivec3( dimf.x, dimf.y, dimf.z );
        grid.h = h;
        grid.pos = box.min();
        return grid;
    }

    // Expand box by absolute distances
    inline void expandAbs( float d ) { m_min -= vec3(d, d, d); m_max += vec3(d, d, d); }
    inline void expandAbs( const vec3 &d ) { m_min -= d; m_max += d; }

    // Expand box relative to current size
    inline void expandRel( float d ) { vec3 dd = d*(m_max-m_min); m_min -= dd; m_max += dd; }
    inline void expandRel( const vec3 &d ) { vec3 dd = d*(m_max-m_min); m_min -= dd; m_max += dd; }

    // Merge two bounding boxes
    inline BBox& operator += ( const BBox &rhs )
    {
        m_min = vec3::min( m_min, rhs.m_min );
        m_max = vec3::max( m_max, rhs.m_max );
        return *this;
    }

    inline BBox  operator +  ( const BBox &rhs ) const
    {
        return BBox( vec3::min(m_min,rhs.m_min), vec3::max(m_max,rhs.m_max) );
    }

    // Incorporate point into bounding box
    inline BBox& operator += ( const vec3 &rhs )
    {
        m_min = vec3::min( m_min, rhs );
        m_max = vec3::max( m_max, rhs );
        return *this;
    }

    inline BBox  operator +  ( const vec3 &rhs ) const
    {
        return BBox( vec3::min(m_min,rhs), vec3::max(m_max,rhs) );
    }

    virtual void render();

    virtual BBox getBBox( const glm::mat4 &ctm )
    {
        BBox box;
        vec3 corner;
        for ( int x = 0, index = 0; x <= 1; x++ ) {
            corner.x = (x?m_max:m_min).x;
            for ( int y = 0; y <= 1; y++ ) {
                corner.y = (y?m_max:m_min).y;
                for ( int z = 0; z <= 1; z++, index++ ) {
                    corner.z = (z?m_max:m_min).z;
                    glm::vec4 point = ctm * glm::vec4(corner.x, corner.y, corner.z, 1.f);
                    box += vec3( point.x, point.y, point.z );
                }
            }
        }
        return box;
    }

    virtual vec3 getCentroid( const glm::mat4 &ctm )
    {
        vec3 c = center();
        glm::vec4 p = ctm * glm::vec4( c.x, c.y, c.z, 1.f );
        return vec3( p.x, p.y, p.z );
    }

};

#endif // BBOX_H
