/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   max.cu
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef MAX_CU
#define MAX_CU

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "cuda/decomposition.cu"
#include "cuda/weighting.cu"

#define CUDA_INCLUDE
#include "sim/particlegrid.h"
#include "sim/particle.h"

#define VEC2IVEC( V ) ( glm::ivec3((int)V.x, (int)V.y, (int)V.z) )

#define CLAMP( X, A, B ) ( (X < A) ? A : ((X > B) ? B : X) )

// Computes M += v * w^T
 __device__ inline void addOuterProduct( const glm::vec3 &v, const glm::vec3 &w, glm::mat3 &M )
{
    M[0][0] += v.x*w.x;    M[1][0] += v.x*w.y;    M[2][0] += v.x*w.z;
    M[0][1] += v.y*w.x;    M[1][1] += v.y*w.y;    M[2][1] += v.y*w.z;
    M[0][2] += v.z*w.x;    M[1][2] += v.z*w.y;    M[2][2] += v.z*w.z;
}

__host__ __device__ void print( const glm::vec3 &v );

// Use weighting functions to compute particle velocity gradient and update particle velocity
__device__ void processGridVelocities( Particle &particle, const Grid &grid, const ParticleGrid::Node *nodes, glm::mat3 &velocityGradient, float alpha )
{
    const glm::vec3 &pos = particle.position;
    const glm::ivec3 &dim = grid.dim;
    const float h = grid.h;

    // Compute neighborhood of particle in grid
    glm::vec3 gridIndex = (pos - grid.pos) / h,
              gridMax = glm::floor( gridIndex + glm::vec3(2,2,2) ),
              gridMin = glm::ceil( gridIndex - glm::vec3(2,2,2) );
    glm::ivec3 maxIndex = glm::clamp( VEC2IVEC(gridMax), glm::ivec3(0,0,0), grid.dim ),
               minIndex = glm::clamp( VEC2IVEC(gridMin), glm::ivec3(0,0,0), grid.dim );

    // For computing particle velocity gradient:
    //      grad(v_p) = sum( v_i * transpose(grad(w_ip)) ) = [3x3 matrix]
    // For updating particle velocity:
    //      v_PIC = sum( v_i * w_ip )
    //      v_FLIP = v_p + sum( dv_i * w_ip )
    //      v = (1-alpha)*v_PIC _ alpha*v_FLIP
    float w;
    glm::vec3 d, s, wg;
    glm::vec3 v_PIC(0,0,0), dv_FLIP(0,0,0);
    for ( int i = minIndex.x; i <= maxIndex.x; ++i ) {
        d.x = i - gridIndex.x;
        d.x *= ( s.x = ( d.x < 0 ) ? -1.f : 1.f );
        int pageOffset = i*(dim.y+1)*(dim.z+1);
        for ( int j = minIndex.y; j <= maxIndex.y; ++j ) {
            d.y = j - gridIndex.y;
            d.y *= ( s.y = ( d.y < 0 ) ? -1.f : 1.f );
            int rowOffset = pageOffset + j*(dim.z+1);
            for ( int k = minIndex.z; k <= maxIndex.z; ++k ) {
                d.z = k - gridIndex.z;
                d.z *= ( s.z = ( d.z < 0 ) ? -1.f : 1.f );
                weightAndGradient( s, d, w, wg );
                const ParticleGrid::Node &node = nodes[rowOffset+k];
                // Particle velocities
                v_PIC += node.velocity * w;
                dv_FLIP += node.velocityChange * w;
                // Velocity gradient
                addOuterProduct( node.velocity, wg, velocityGradient );
            }
        }
    }

    particle.velocity = (1.f-alpha)*v_PIC + alpha*(particle.velocity+dv_FLIP);
}

__device__ void updateParticleDeformationGradients( Particle &particle, const glm::mat3 &velocityGradient, float timeStep,
                                                    float criticalCompression, float criticalStretch )
{
    // Temporarily assign all deformation to elastic portion
    glm::mat3 F = (glm::mat3(1.f) + timeStep*velocityGradient) * particle.elasticF;

    // Clamp the singular values
    glm::mat3 W, S, Sinv, V;
    computeSVD( F, W, S, V );
    S = glm::mat3( CLAMP( S[0][0], criticalCompression, criticalStretch ), 0.f, 0.f,
                   0.f, CLAMP( S[1][1], criticalCompression, criticalStretch ), 0.f,
                   0.f, 0.f, CLAMP( S[2][2], criticalCompression, criticalStretch ) );
    Sinv = glm::mat3( 1.f/S[0][0], 0.f, 0.f,
                      0.f, 1.f/S[1][1], 0.f,
                      0.f, 0.f, 1.f/S[2][2] );

    // Compute final deformation components
    particle.elasticF = W * S * glm::transpose(V);
    particle.plasticF = V * Sinv * glm::transpose(W) * particle.plasticF;
}

__device__ void checkForCollisions( Particle &particle )
{

}

// NOTE: assumes particleCount % blockDim.x = 0, so tid is never out of range!
// criticalCompression = 1 - theta_c
// criticalStretch = 1 + theta_s
__global__ void updateParticlesFromGrid( Particle *particles, const ParticleGrid grid, const ParticleGrid::Node *nodes, float timeStep,
                                         float criticalCompression, float criticalStretch, float alpha )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    Particle &particle = particles[tid];

    // Update particle velocities and fill in velocity gradient for deformation gradient computation
    glm::mat3 velocityGradient = glm::mat3( 0.f );
    processGridVelocities( particle, grid, nodes, velocityGradient, alpha );

    updateParticleDeformationGradients( particle, velocityGradient, timeStep, criticalCompression, criticalStretch );

    checkForCollisions( particle );

    particle.position += timeStep * particle.velocity;
}

extern "C" { void grid2ParticlesTests(); }

__host__ __device__ void print( const glm::vec3 &v )
{
    printf( "    %10.5f    %10.5f    %10.5f\n", v.x, v.y, v.z );
}

__host__ __device__ void print( const glm::mat3 &M )
{
    for ( int i = 0; i < 3; ++i ) {
        for ( int j = 0; j < 3; j++ ) {
            printf( "    %10.5f", M[j][i] );
        }
        printf( "\n" );
    }
}

__host__ __device__ void print( const glm::mat4 &M )
{
    for ( int i = 0; i < 4; ++i ) {
        for ( int j = 0; j < 4; j++ ) {
            printf( "    %10.5f", M[j][i] );
        }
        printf( "\n" );
    }
}

#include "glm/gtc/epsilon.hpp"

__host__ __device__ bool equals( const glm::mat3 &A, const glm::mat3 &B )
{
    for ( int i = 0; i < 3; ++i ) {
        for ( int j = 0; j < 3; ++j ) {
            if ( glm::epsilonNotEqual(A[j][i], B[j][i], (float)EPSILON) ) {
                return false;
            }
        }
    }
    return true;
}

#include "glm/gtc/matrix_transform.hpp"

__global__ void outerProductTestKernel()
{
    {
        printf( "Outer Product Test 1: " );

        glm::vec3 v( 1, 2, 3 ), w( 4, 5, 6 );
        glm::mat3 M( 0.f );
        addOuterProduct( v, w, M );

        glm::mat3 expected = glm::transpose(glm::mat3(4, 5, 6, 8, 10, 12, 12, 15, 18));
        if ( !equals(M, expected) ) {
            printf( "FAILED:\n" );
            printf( "    Expected matrix:\n" ); print( expected );
            printf( "    Got matrix:\n" ); print( M );
        } else {
            printf( "PASSED.\n" );
        }
    }

    {
        printf( "Outer Product Test 2: " );

        glm::vec3 v( 3, 2, 1 ), w( 10, 9, 10 );
        glm::mat3 M( 0.f );
        addOuterProduct( v, w, M );

        glm::mat3 expected = glm::transpose(glm::mat3(30, 27, 30, 20, 18, 20, 10, 9, 10));
        if ( !equals(M, expected) ) {
            printf( "FAILED:\n" );
            printf( "    Expected matrix:\n" ); print( expected );
            printf( "    Got matrix:\n" ); print( M );
        } else {
            printf( "PASSED.\n" );
        }
    }

    {
        printf( "Outer Product Test 3: " );

        glm::vec3 v( 10, 5, 1 ), w( 8, 2, 9 );
        glm::mat3 M( 0.f );
        addOuterProduct( v, w, M );

        glm::mat3 expected = glm::transpose(glm::mat3(80, 20, 90, 40, 10, 45, 8, 2, 9));
        if ( !equals(M, expected) ) {
            printf( "FAILED:\n" );
            printf( "    Expected matrix:\n" ); print( expected );
            printf( "    Got matrix:\n" ); print( M );
        } else {
            printf( "PASSED.\n" );
        }
    }

}

__global__ void velocityTest( Particle *particles, glm::mat3 *gradients, const Grid grid, ParticleGrid::Node *nodes, float alpha )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Particle &particle = particles[tid];
    glm::mat3 &velocityGradient = gradients[tid];
    velocityGradient = glm::mat3(0.f);
    processGridVelocities( particle, grid, nodes, velocityGradient, alpha );
}

#include "glm/gtc/random.hpp"
#include "glm/gtc/matrix_access.hpp"
#include "common/common.h"
#include "common/math.h"

__host__ void grid2ParticlesTests()
{

    outerProductTestKernel<<<1,1>>>();
    cudaDeviceSynchronize();

    printf( "Velocity Processing Test:\n" );

    const int dim = 8;
    ParticleGrid grid;
    grid.dim = glm::ivec3( dim, dim, dim );
    grid.h = 1.f/dim;
    grid.pos = glm::vec3(0,0,0);

    int nParticles = dim*dim*dim;
    printf( "    Generating %d particles (%.2f MB)...\n",
            nParticles, nParticles*sizeof(Particle)/1e6 );
    fflush(stdout);
    Particle *particles = new Particle[nParticles];
    for ( int i = 0; i < dim; ++i ) {
        for ( int j = 0; j < dim; ++j ) {
            for ( int k = 0; k < dim; ++k ) {
                Particle particle;
                particle.position = grid.pos + grid.h*glm::vec3( i+0.5f, j+0.5f, k+0.5f );
                particle.velocity = glm::vec3( 0.f, -0.124f, 0.f );
                particle.elasticF = glm::mat3(1.f);
                particle.plasticF = glm::mat3(1.f);
                particles[i*dim*dim+j*dim+k] = particle;
            }
        }
    }

    printf( "    Generating %d grid nodes (%.2f MB)...\n",
            (dim+1)*(dim+1)*(dim+1), (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGrid::Node)/1e6 );
    fflush(stdout);
    ParticleGrid::Node *nodes = grid.createNodes();
    for ( int i = 0; i <= dim; ++i ) {
        for ( int j = 0; j <= dim; ++j ) {
            for ( int k = 0; k <= dim; ++k ) {
                ParticleGrid::Node node;
                node.velocity = glm::vec3( 0.f, -0.125f, 0.f );
                node.velocityChange = glm::vec3( 0.f, -0.001f, 0.f );
                nodes[i*(dim+1)*(dim+1)+j*(dim+1)+k] = node;
            }
        }
    }

    printf( "    Allocating kernel resources...\n" ); fflush(stdout);
    Particle *devParticles;
    ParticleGrid::Node *devNodes;
    glm::mat3 *devGradients;
    cudaError_t error;
    error = cudaMalloc( &devParticles, nParticles*sizeof(Particle) );
    error = cudaMemcpy( devParticles, particles, nParticles*sizeof(Particle), cudaMemcpyHostToDevice );
    error = cudaMalloc( &devNodes, (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGrid::Node) );
    error = cudaMemcpy( devNodes, nodes, (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGrid::Node), cudaMemcpyHostToDevice );
    error = cudaMalloc( &devGradients, nParticles*sizeof(glm::mat3) );

    if ( error != cudaSuccess ) {
        printf( "    FAILED: %s\n", _cudaGetErrorEnum(error) );
    } else {
        static const int blockSize = 128;
        for ( int i = 0; i < 5; ++i ) {
            TIME( "    Launching kernel... ", "Kernel finished.\n",
                  velocityTest<<< (nParticles+blockSize-1)/blockSize, blockSize >>>( devParticles, devGradients, grid, devNodes, 0.95f );
                  error = cudaDeviceSynchronize();
            );
            if ( error != cudaSuccess ) {
                printf( "    FAILED: %s\n", _cudaGetErrorEnum(error) );
                break;
            }
            fflush(stdout);
        }
    }

    error = cudaMemcpy( particles, devParticles, nParticles*sizeof(Particle), cudaMemcpyDeviceToHost );

    // Velocity checks
    float expected[] = { -0.127181205566406f, -0.127181205566406f, -0.127845724062500f, -0.128524381250000f };
    if ( NEQF(particles[64].velocity.y, expected[0] ) ) {
        printf( "    FAILED: expected particles[64].velocity.y = %.6g, got %.6g\n", expected[0], particles[64].velocity.y );
    } else if ( NEQF(particles[120].velocity.y, expected[1] ) ) {
        printf( "    FAILED: expected particles[120].velocity.y = %.6g, got %.6g\n", expected[1], particles[120].velocity.y );
    } else if ( NEQF(particles[167].velocity.y, expected[2] ) ) {
        printf( "    FAILED: expected particles[167].velocity.y = %.6g, got %.6g\n", expected[2], particles[167].velocity.y );
    } else if ( NEQF(particles[394].velocity.y, expected[3] ) ) {
        printf( "    FAILED: expected particles[394].velocity.y = %.6g, got %.6g\n", expected[3], particles[394].velocity.y );
    } else {
        printf( "    ALL VELOCITY CHECKS PASSED.\n" );
    }

    glm::mat3 *gradients = new glm::mat3[nParticles];
    error = cudaMemcpy( gradients, devGradients, nParticles*sizeof(glm::mat3), cudaMemcpyDeviceToHost );

    bool fail = false;
    for ( int i = 0; i < nParticles; ++i ) {
        glm::vec3 row = glm::row( gradients[i], 1 );
        if ( i >= 64 && i < 448 ) {
            if ( NEQF(row.x, 0.f) ) {
                printf( "    FAILED: expected particles[%d] x-velocity gradient to be ~0, got %f\n", i, row.x );
                fail = true;
                break;
            }
        }
        if ( i >= 65 && i < 386 && (i-65) % 64 == 0 ) {
            if ( NEQF(row.y, 0.015625f) ) {
                printf( "    FAILED: expected particles[%d] y-velocity gradient to be 0.015625, got %f\n", i, row.y );
                fail = true;
                break;
            }
        }
        if ( i >= 121 && i < 442 && (i-121) % 64 == 0 ) {
            if ( NEQF(row.y, -0.015625f ) ) {
                printf( "    FAILED: expected particles[%d] y-velocity gradient to be -0.015299, got %f\n", i, row.y );
                fail = true;
                break;
            }
        }
        if ( i >= 30 && (i-30) % 64 == 0 ) {
            if ( NEQF(row.y, 0.f ) ) {
                printf( "    FAILED: expected particles[%d] y-velocity gradient to be 0, got %f\n", i, row.y );
                fail = true;
                break;
            }
        }
        if ( i == nParticles-1 ) {
            glm::mat3 expected = glm::mat3(0.f,-0.014980740017361f,0.f,0.f,-0.014980740017361f,0.f,0.f,-0.014980740017361f,0.f);
            if ( !equals(gradients[i], expected) ) {
                printf( "    FAILED: expected:\n" );
                print(expected);
                printf( "            got: \n" );
                print( gradients[i] );
            }
        }
    }

    if ( !fail ) printf( "    ALL VELOCITY GRADIENT CHECKS PASSED.\n" );

    printf( "    Freeing kernel resources...\n" ); fflush(stdout);
    cudaFree( devParticles );
    cudaFree( devNodes );
    cudaFree( devGradients );
    delete [] particles;
    delete [] nodes;
    delete [] gradients;

    printf( "Done.\n" );
}

#endif // MAX_CU
