/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   matrix.cu
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 15 Apr 2014
**
**************************************************************************/

#ifndef MATRIX_CU
#define MATRIX_CU

#include <cuda.h>
#include <cuda_runtime.h>

#include "common/math.h"

#define GLM_FORCE_RADIANS
#include "glm/mat3x3.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "cuda/vector.cu"

// Optimize C = glm::transpose(A) * B
__device__ __forceinline__ void multiplyTransposeL( const glm::mat3 &A, const glm::mat3 &B, glm::mat3 &C )
{
    glm::mat3 _C;
    float *c = glm::value_ptr(_C);
    const float *a = glm::value_ptr(A);
    const float *b = glm::value_ptr(B);
    c[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    c[1] = a[3]*b[0] + a[4]*b[1] + a[5]*b[2];
    c[2] = a[6]*b[0] + a[7]*b[1] + a[8]*b[2];
    c[3] = a[0]*b[3] + a[1]*b[4] + a[2]*b[5];
    c[4] = a[3]*b[3] + a[4]*b[4] + a[5]*b[5];
    c[5] = a[6]*b[3] + a[7]*b[4] + a[8]*b[5];
    c[6] = a[0]*b[6] + a[1]*b[7] + a[2]*b[8];
    c[7] = a[3]*b[6] + a[4]*b[7] + a[5]*b[8];
    c[8] = a[6]*b[6] + a[7]*b[7] + a[8]*b[8];
    C = _C;
}

__device__ __forceinline__ void multiply( const glm::mat3 &A, const glm::mat3 &B, glm::mat3 &C )
{
    glm::mat3 _C;
    float *c = glm::value_ptr(_C);
    const float *a = glm::value_ptr(A);
    const float *b = glm::value_ptr(B);
    c[0] = a[0]*b[0] + a[3]*b[1] + a[6]*b[2];
    c[1] = a[1]*b[0] + a[4]*b[1] + a[7]*b[2];
    c[2] = a[2]*b[0] + a[5]*b[1] + a[8]*b[2];
    c[3] = a[0]*b[3] + a[3]*b[4] + a[6]*b[5];
    c[4] = a[1]*b[3] + a[4]*b[4] + a[7]*b[5];
    c[5] = a[2]*b[3] + a[5]*b[4] + a[8]*b[5];
    c[6] = a[0]*b[6] + a[3]*b[7] + a[6]*b[8];
    c[7] = a[1]*b[6] + a[4]*b[7] + a[7]*b[8];
    c[8] = a[2]*b[6] + a[5]*b[7] + a[8]*b[8];
    C = _C;
}

struct mat3
{
    float data[9];

    __host__ __device__ __forceinline__
    mat3( float f = 1.f )
    {
        data[0] = f; data[3] = 0; data[6] = 0;
        data[1] = 0; data[4] = f; data[7] = 0;
        data[2] = 0; data[5] = 0; data[8] = f;
    }

    __host__ __device__ __forceinline__
    mat3( float a, float b, float c, float d, float e, float f, float g, float h, float i )
    {
        data[0] = a; data[3] = d; data[6] = g;
        data[1] = b; data[4] = e; data[7] = h;
        data[2] = c; data[5] = f; data[8] = i;
    }

    __host__ __device__ __forceinline__
    mat3( const glm::mat3 &M )
    {
        const float *m = glm::value_ptr(M);
        data[0] = m[0]; data[3] = m[3]; data[6] = m[6];
        data[1] = m[1]; data[4] = m[4]; data[7] = m[7];
        data[2] = m[2]; data[5] = m[5]; data[8] = m[8];
    }

    __host__ __device__ __forceinline__
    mat3( const vec3 &c0, const vec3 &c1, const vec3 &c2 )
    {
        data[0] = c0.x; data[3] = c1.x; data[6] = c2.x;
        data[1] = c0.y; data[4] = c1.y; data[7] = c2.y;
        data[2] = c0.z; data[5] = c1.z; data[8] = c2.z;
    }

    __host__ __device__  __forceinline__
    mat3& operator = ( const mat3 &rhs )
    {
        data[0] = rhs[0]; data[3] = rhs[3]; data[6] = rhs[6];
        data[1] = rhs[1]; data[4] = rhs[4]; data[7] = rhs[7];
        data[2] = rhs[2]; data[5] = rhs[5]; data[8] = rhs[8];
        return *this;
    }

    __host__ __device__ __forceinline__
    mat3& operator = ( const glm::mat3 &rhs )
    {
        const float *r = glm::value_ptr(rhs);
        data[0] = r[0]; data[3] = r[3]; data[6] = r[6];
        data[1] = r[1]; data[4] = r[4]; data[7] = r[7];
        data[2] = r[2]; data[5] = r[5]; data[8] = r[8];
        return *this;
    }

    __host__ __device__ static bool equals( const mat3 &A, const mat3 &B )
    {
        for ( int i = 0; i < 9; ++i ) {
            if ( NEQF(A[0], B[0]) ) {
                return false;
            }
        }
        return true;
    }

    __host__ __device__ __forceinline__
    glm::mat3 toGLM() const
    {
        return glm::mat3( data[0], data[1], data[2],
                          data[3], data[4], data[5],
                          data[6], data[7], data[8] );
    }

    __host__ __device__ __forceinline__
    static mat3 outerProduct( const vec3 &v, const vec3& w )
    {
        return mat3( v.x*w.x, v.y*w.x, v.z*w.x,
                     v.x*w.y, v.y*w.y, v.z*w.y,
                     v.x*w.z, v.y*w.z, v.z*w.z );
    }

    __host__ __device__ __forceinline__ float& operator [] ( int i ) { return data[i]; }
    __host__ __device__ __forceinline__ float  operator [] ( int i ) const { return data[i]; }

    __host__ __device__ __forceinline__
    vec3 row( int i ) const { return vec3( data[i], data[i+3], data[i+6] ); }

    __host__ __device__ __forceinline__
    vec3 column( int i ) const { int j = 3*i; return vec3( data[j], data[j+1], data[j+2] ); }

    __host__ __device__ __forceinline__
    mat3& operator *= ( const mat3 &rhs )
    {
        mat3 tmp;
        tmp[0] = data[0]*rhs[0] + data[3]*rhs[1] + data[6]*rhs[2];
        tmp[1] = data[1]*rhs[0] + data[4]*rhs[1] + data[7]*rhs[2];
        tmp[2] = data[2]*rhs[0] + data[5]*rhs[1] + data[8]*rhs[2];
        tmp[3] = data[0]*rhs[3] + data[3]*rhs[4] + data[6]*rhs[5];
        tmp[4] = data[1]*rhs[3] + data[4]*rhs[4] + data[7]*rhs[5];
        tmp[5] = data[2]*rhs[3] + data[5]*rhs[4] + data[8]*rhs[5];
        tmp[6] = data[0]*rhs[6] + data[3]*rhs[7] + data[6]*rhs[8];
        tmp[7] = data[1]*rhs[6] + data[4]*rhs[7] + data[7]*rhs[8];
        tmp[8] = data[2]*rhs[6] + data[5]*rhs[7] + data[8]*rhs[8];
        return (*this = tmp);
    }

    __host__ __device__ __forceinline__
    mat3 operator * ( const mat3 &rhs ) const
    {
        mat3 result;
        result[0] = data[0]*rhs[0] + data[3]*rhs[1] + data[6]*rhs[2];
        result[1] = data[1]*rhs[0] + data[4]*rhs[1] + data[7]*rhs[2];
        result[2] = data[2]*rhs[0] + data[5]*rhs[1] + data[8]*rhs[2];
        result[3] = data[0]*rhs[3] + data[3]*rhs[4] + data[6]*rhs[5];
        result[4] = data[1]*rhs[3] + data[4]*rhs[4] + data[7]*rhs[5];
        result[5] = data[2]*rhs[3] + data[5]*rhs[4] + data[8]*rhs[5];
        result[6] = data[0]*rhs[6] + data[3]*rhs[7] + data[6]*rhs[8];
        result[7] = data[1]*rhs[6] + data[4]*rhs[7] + data[7]*rhs[8];
        result[8] = data[2]*rhs[6] + data[5]*rhs[7] + data[8]*rhs[8];
        return result;
    }

    __host__ __device__ __forceinline__
    mat3& operator += ( const mat3 &rhs )
    {
        data[0] += rhs[0]; data[3] += rhs[3]; data[6] += rhs[6];
        data[1] += rhs[1]; data[4] += rhs[4]; data[7] += rhs[7];
        data[2] += rhs[2]; data[5] += rhs[5]; data[8] += rhs[8];
        return *this;
    }

    __host__ __device__ __forceinline__
    mat3 operator + ( const mat3 &rhs ) const
    {
        mat3 tmp = *this;
        tmp[0] += rhs[0]; tmp[3] += rhs[3]; tmp[6] += rhs[6];
        tmp[1] += rhs[1]; tmp[4] += rhs[4]; tmp[7] += rhs[7];
        tmp[2] += rhs[2]; tmp[5] += rhs[5]; tmp[8] += rhs[8];
        return tmp;
    }

    __host__ __device__ __forceinline__
    mat3& operator -= ( const mat3 &rhs )
    {
        data[0] -= rhs[0]; data[3] -= rhs[3]; data[6] -= rhs[6];
        data[1] -= rhs[1]; data[4] -= rhs[4]; data[7] -= rhs[7];
        data[2] -= rhs[2]; data[5] -= rhs[5]; data[8] -= rhs[8];
        return *this;
    }

    __host__ __device__ __forceinline__
    mat3 operator - ( const mat3 &rhs ) const
    {
        mat3 tmp = *this;
        tmp[0] -= rhs[0]; tmp[3] -= rhs[3]; tmp[6] -= rhs[6];
        tmp[1] -= rhs[1]; tmp[4] -= rhs[4]; tmp[7] -= rhs[7];
        tmp[2] -= rhs[2]; tmp[5] -= rhs[5]; tmp[8] -= rhs[8];
        return tmp;
    }

    __host__ __device__ __forceinline__
    mat3& operator *= ( float f )
    {
        data[0] *= f; data[3] *= f; data[6] *= f;
        data[1] *= f; data[4] *= f; data[7] *= f;
        data[2] *= f; data[5] *= f; data[8] *= f;
        return *this;
    }

    __host__ __device__ __forceinline__
    mat3 operator * ( float f ) const
    {
        mat3 tmp = *this;
        tmp[0] *= f; tmp[3] *= f; tmp[6] *= f;
        tmp[1] *= f; tmp[4] *= f; tmp[7] *= f;
        tmp[2] *= f; tmp[5] *= f; tmp[8] *= f;
        return tmp;
    }

    __host__ __device__ __forceinline__
    mat3& operator /= ( float f )
    {
        float fi = 1.f/f;
        data[0] *= fi; data[3] *= fi; data[6] *= fi;
        data[1] *= fi; data[4] *= fi; data[7] *= fi;
        data[2] *= fi; data[5] *= fi; data[8] *= fi;
        return *this;
    }

    __host__ __device__ __forceinline__
    mat3 operator / ( float f ) const
    {
        mat3 tmp = *this;
        float fi = 1.f/f;
        tmp[0] *= fi; tmp[3] *= fi; tmp[6] *= fi;
        tmp[1] *= fi; tmp[4] *= fi; tmp[7] *= fi;
        tmp[2] *= fi; tmp[5] *= fi; tmp[8] *= fi;
        return tmp;
    }

    __host__ __device__ __forceinline__
    static mat3 transpose( const mat3 &m )
    {
        return mat3( m[0], m[3], m[6],
                     m[1], m[4], m[7],
                     m[2], m[5], m[8] );
    }

    // Optimize transpose(A) * B;
    __host__ __device__ __forceinline__
    static mat3 multiplyTransposeL( const mat3 &A, const mat3 &B )
    {
        mat3 tmp;
        tmp[0] = A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
        tmp[1] = A[3]*B[0] + A[4]*B[1] + A[5]*B[2];
        tmp[2] = A[6]*B[0] + A[7]*B[1] + A[8]*B[2];
        tmp[3] = A[0]*B[3] + A[1]*B[4] + A[2]*B[5];
        tmp[4] = A[3]*B[3] + A[4]*B[4] + A[5]*B[5];
        tmp[5] = A[6]*B[3] + A[7]*B[4] + A[8]*B[5];
        tmp[6] = A[0]*B[6] + A[1]*B[7] + A[2]*B[8];
        tmp[7] = A[3]*B[6] + A[4]*B[7] + A[5]*B[8];
        tmp[8] = A[6]*B[6] + A[7]*B[7] + A[8]*B[8];
        return tmp;
    }

    __host__ __device__ __forceinline__
    static mat3 multiplyTransposeR( const mat3 &A, const mat3 &B )
    {
        return A * transpose(B);
    }

};

__host__ __device__ __forceinline__
mat3 operator * ( float f, const mat3 &m ) { return m*f; }

#endif // MATRIX_CU
