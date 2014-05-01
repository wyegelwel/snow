/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   vector.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 15 Apr 2014
**
**************************************************************************/

#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#ifndef GLM_FORCE_RADIANS
    #define GLM_FORCE_RADIANS
#endif
#include "glm/vec3.hpp"

#include "common/math.h"

struct vec3
{

    float3 data;

    __host__ __device__ __forceinline__ float& x() { return data.x; }
    __host__ __device__ __forceinline__ float x() const { return data.x; }
    __host__ __device__ __forceinline__ float& y() { return data.y; }
    __host__ __device__ __forceinline__ float y() const { return data.y; }
    __host__ __device__ __forceinline__ float& z() { return data.z; }
    __host__ __device__ __forceinline__ float z() const { return data.z; }

    __host__ __device__ __forceinline__ float* ptr() { return &data.x; }
    __host__ __device__ __forceinline__ const float* ptr() const { return &data.x; }

    __host__ __device__ __forceinline__
    vec3() { data.x = 0.f; data.y = 0.f; data.z = 0.f; }

    __host__ __device__ __forceinline__
    vec3( float v ) { data.x = v; data.y = v; data.z = v; }

    __host__ __device__ __forceinline__
    vec3( float xx, float yy, float zz ) { data.x = xx; data.y = yy; data.z = zz; }

    __host__ __device__ __forceinline__
    vec3( const vec3 &v ) { data = v.data; }

    __host__ __device__ __forceinline__
    vec3( const glm::vec3 &v ) { data.x = v.x; data.y = v.y; data.z = v.z; }

    __host__ __device__ __forceinline__
    vec3( const glm::ivec3 &v ) { data.x = (float)v.x; data.y = (float)v.y; data.z = (float)v.z; }

    __host__ __device__ __forceinline__
    operator glm::vec3() const { return glm::vec3( data.x, data.y, data.z ); }

    __host__ __device__ __forceinline__
    operator glm::ivec3() const { return glm::ivec3( (int)data.x, (int)data.y, (int)data.z ); }

    __host__ __device__ __forceinline__
    vec3& operator = ( const vec3 &rhs ) { data = rhs.data; return *this; }

    __host__ __device__ __forceinline__
    vec3& operator = ( const glm::vec3 &rhs ) { data.x = rhs.x; data.y = rhs.y; data.z = rhs.z; return *this; }

    __host__ __device__ __forceinline__
    vec3& operator = ( const glm::ivec3 &rhs ) { data.x = (float)rhs.x; data.y = (float)rhs.y; data.z = (float)rhs.z; return *this; }

    __host__ __device__ __forceinline__
    int majorAxis() { return ( (fabsf(data.x)>fabsf(data.y)) ? ((fabsf(data.x)>fabsf(data.z)) ? 0 : 2) : ((fabsf(data.y)>fabsf(data.z)) ? 1 : 2) ); }

    __host__ __device__ __forceinline__
    float& operator [] ( int i ) { return *(&data.x+i); }

    __host__ __device__ __forceinline__
    float operator [] ( int i ) const { return *(&data.x+i); }

    __host__ __device__ __forceinline__
    static float dot( const vec3 &a, const vec3 &b ) { return a.data.x*b.data.x + a.data.y*b.data.y + a.data.z*b.data.z; }

    __host__ __device__ __forceinline__
    static vec3 cross( const vec3 &a, const vec3 &b )
    {
        return vec3( a.data.y*b.data.z - a.data.z*b.data.y,
                     a.data.z*b.data.x - a.data.x*b.data.z,
                     a.data.x*b.data.y - a.data.y*b.data.x );
    }

    __host__ __device__ __forceinline__
    static vec3 floor( const vec3 &v ) { return vec3( floorf(v.data.x), floorf(v.data.y), floorf(v.data.z) ); }

    __host__ __device__ __forceinline__
    static vec3 ceil( const vec3 &v ) { return vec3( ceilf(v.data.x), ceilf(v.data.y), ceilf(v.data.z) ); }

    __host__ __device__ __forceinline__
    static vec3 abs( const vec3 &v ) { return vec3( fabsf(v.data.x), fabsf(v.data.y), fabsf(v.data.z) ); }

    __host__ __device__ __forceinline__
    static vec3 round( const vec3 &v ) { return vec3( roundf(v.data.x), roundf(v.data.y), roundf(v.data.z) ); }

#define SIGN(X) ((0<(X)-((X)<0)))
    //From http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
    __host__ __device__ __forceinline__
    static vec3 sign( const vec3 &v ) { return vec3(SIGN(v.data.x), SIGN(v.data.y), SIGN(v.data.z) );}
#undef SIGN

    __host__ __device__ __forceinline__
    static vec3 min( const vec3 &v, const vec3 &w ) { return vec3( fminf(v.data.x, w.data.x), fminf(v.data.y, w.data.y), fminf(v.data.z,w.data.z) ); }

    __host__ __device__ __forceinline__
    static vec3 max( const vec3 &v, const vec3 &w ) { return vec3( fmaxf(v.data.x, w.data.x), fmaxf(v.data.y, w.data.y), fmaxf(v.data.z,w.data.z) ); }

    __host__ __device__ __forceinline__
    static vec3 mix(const vec3 &v, const vec3 &w, const vec3 &a) { return vec3(v.data.x*(1.f-a.data.x)+w.data.x*a.data.x, v.data.y*(1.f-a.data.y)+w.data.y*a.data.y, v.data.z*(1.f-a.data.z)+w.data.z*a.data.z); }

    __host__ __device__ __forceinline__
    static vec3 mix(const vec3 &v, const vec3 &w, float a) { return vec3(v.data.x*(1.f-a)+w.data.x*a, v.data.y*(1.f-a)+w.data.y*a, v.data.z*(1.f-a)+w.data.z*a); }

    __host__ __device__ __forceinline__
    static float length2( const vec3 &v ) { return v.data.x*v.data.x + v.data.y*v.data.y + v.data.z*v.data.z; }

    __host__ __device__ __forceinline__
    static float length( const vec3 &v ) { return sqrtf( v.data.x*v.data.x + v.data.y*v.data.y + v.data.z*v.data.z ); }

    __host__ __device__ __forceinline__
    static vec3 normalize( const vec3 &v ) { float f = 1.f/sqrtf(v.data.x*v.data.x+v.data.y*v.data.y+v.data.z*v.data.z); return vec3( f*v.data.x, f*v.data.y, f*v.data.z ); }

    __host__ __device__ __forceinline__
    vec3& mult (float f ) { data.x *= f; data.y *= f; data.z *= f; return *this;}

    __host__ __device__ __forceinline__
    vec3& add (float f ) { data.x += f; data.y += f; data.z += f; return *this;}

    __host__ __device__ __forceinline__
    vec3& add (const vec3 &v ) { data.x += v.data.x; data.y += v.data.y; data.z += v.data.z; return *this;}

    __host__ __device__ __forceinline__
    vec3& operator += ( const vec3 &rhs ) { data.x += rhs.data.x; data.y += rhs.data.y; data.z += rhs.data.z; return *this; }

    __host__ __device__ __forceinline__
    vec3 operator + ( const vec3 &rhs ) const { return vec3( data.x+rhs.data.x, data.y+rhs.data.y, data.z+rhs.data.z ); }

    __host__ __device__ __forceinline__
    vec3& operator -= ( const vec3 &rhs ) { data.x -= rhs.data.x; data.y -= rhs.data.y; data.z -= rhs.data.z; return *this; }

    __host__ __device__ __forceinline__
    vec3 operator - ( const vec3 &rhs ) const { return vec3( data.x-rhs.data.x, data.y-rhs.data.y, data.z-rhs.data.z ); }

    __host__ __device__ __forceinline__
    vec3& operator *= ( const vec3 &rhs ) { data.x *= rhs.data.x; data.y *= rhs.data.y; data.z *= rhs.data.z; return *this; }

    __host__ __device__ __forceinline__
    vec3 operator * ( const vec3 &rhs ) const { return vec3( data.x*rhs.data.x, data.y*rhs.data.y, data.z*rhs.data.z ); }

    __host__ __device__ __forceinline__
    vec3& operator /= ( const vec3 &rhs ) { data.x /= rhs.data.x; data.y /= rhs.data.y; data.z /= rhs.data.z; return *this; }

    __host__ __device__ __forceinline__
    vec3 operator / ( const vec3 &rhs ) const { return vec3( data.x/rhs.data.x, data.y/rhs.data.y, data.z/rhs.data.z ); }

    __host__ __device__ __forceinline__
    vec3& operator *= ( float f )  { data.x *= f; data.y *= f; data.z *= f; return *this; }

    __host__ __device__ __forceinline__
    vec3 operator * ( float f ) const { return vec3( f*data.x, f*data.y, f*data.z ); }

    __host__ __device__ __forceinline__
    vec3& operator *= ( double d )  { data.x = (float)(data.x*d); data.y = (float)(data.y*d); data.z = (float)(data.z*d); return *this; }

    __host__ __device__ __forceinline__
    vec3 operator * ( double d ) const { return vec3( (float)(data.x*d), (float)(data.y*d), (float)(data.z*d) ); }

    __host__ __device__ __forceinline__
    vec3& operator /= ( float f ) { float fi = 1./f; data.x *= fi; data.y *= fi; data.z *= fi; return *this; }

    __host__ __device__ __forceinline__
    vec3 operator / ( float f ) const { float fi = 1.f/f; return vec3( data.x*fi, data.y*fi, data.z*fi ); }

    __host__ __device__ __forceinline__
    vec3& operator += ( float f ) { data.x += f; data.y += f; data.z += f; return *this; }

    __host__ __device__ __forceinline__
    vec3 operator + ( float f ) const { return vec3( data.x+f, data.y+f, data.z+f ); }

    __host__ __device__ __forceinline__
    vec3& operator -= ( float f ) { data.x -= f; data.y -= f; data.z -= f; return *this; }

    __host__ __device__ __forceinline__
    vec3 operator - ( float f ) const { return vec3( data.x-f, data.y-f, data.z-f ); }

    __host__ __device__ __forceinline__
    bool valid( bool *nan = NULL ) const
    {
        if ( __isnanf(data.x) || __isnanf(data.y) || __isnanf(data.z) ) {
            if ( nan ) *nan = true;
            return false;
        } else if ( __isinff(data.x) || __isinff(data.y) || __isinff(data.z) ) {
            if ( nan ) *nan = false;
            return false;
        }
        return true;
    }

    __host__ __device__ __forceinline__
    static void print( const vec3 &v )
    {
        printf( "[%10f %10f %10f]\n", v.data.x, v.data.y, v.data.z );
    }

    __host__ __device__ __forceinline__
    bool operator == ( const vec3 &v ) const
    {
        return EQF( data.x, v.data.x ) && EQF( data.y, v.data.y ) && EQF( data.z, v.data.z );
    }

    __host__ __device__ __forceinline__
    bool operator != ( const vec3 &v ) const
    {
        return NEQF( data.x, v.data.x ) || NEQF( data.y, v.data.y ) || NEQF( data.z, v.data.z );
    }

};

__host__ __device__ __forceinline__
vec3 operator - ( const vec3 &v ) { return vec3( -v.data.x, -v.data.y, -v.data.z ); }

__host__ __device__ __forceinline__
vec3 operator * ( float f, const vec3 &v ) { return vec3( f*v.data.x, f*v.data.y, f*v.data.z ); }

__host__ __device__ __forceinline__
vec3 operator * ( double f, const vec3 &v ) { return vec3( (float)(f*v.data.x), (float)(f*v.data.y), (float)(f*v.data.z) ); }

#endif // VECTOR_H
