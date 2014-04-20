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

#include "glm/gtc/matrix_access.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "common/common.h"
#include "common/math.h"

#define CUDA_INCLUDE
#include "sim/particlegridnode.h"
#include "sim/particle.h"

#include "cuda/tim.cu"

#define VEC2IVEC( V ) ( glm::ivec3((int)V.x, (int)V.y, (int)V.z) )

#define CLAMP( X, A, B ) ( (X < A) ? A : ((X > B) ? B : X) )

// Use weighting functions to compute particle velocity gradient and update particle velocity
__device__ void processGridVelocities( Particle &particle, const Grid &grid, const ParticleGridNode *nodes, mat3 &velocityGradient, float alpha )
{
    const vec3 &pos = particle.position;
    const glm::ivec3 &dim = grid.dim;
    const float h = grid.h;

    // Compute neighborhood of particle in grid
    vec3 gridIndex = (pos - grid.pos) / h,
         gridMax = vec3::floor( gridIndex + vec3(2,2,2) ),
         gridMin = vec3::ceil( gridIndex - vec3(2,2,2) );
    glm::ivec3 maxIndex = glm::clamp( VEC2IVEC(gridMax), glm::ivec3(0,0,0), grid.dim ),
               minIndex = glm::clamp( VEC2IVEC(gridMin), glm::ivec3(0,0,0), grid.dim );

    // For computing particle velocity gradient:
    //      grad(v_p) = sum( v_i * transpose(grad(w_ip)) ) = [3x3 matrix]
    // For updating particle velocity:
    //      v_PIC = sum( v_i * w_ip )
    //      v_FLIP = v_p + sum( dv_i * w_ip )
    //      v = (1-alpha)*v_PIC _ alpha*v_FLIP
    vec3 v_PIC(0,0,0), dv_FLIP(0,0,0);
    int rowSize = dim.z+1;
    int pageSize = (dim.y+1)*rowSize;
    for ( int i = minIndex.x; i <= maxIndex.x; ++i ) {
        vec3 d, s;
        d.x = i - gridIndex.x;
        d.x *= ( s.x = ( d.x < 0 ) ? -1.f : 1.f );
        int pageOffset = i*pageSize;
        for ( int j = minIndex.y; j <= maxIndex.y; ++j ) {
            d.y = j - gridIndex.y;
            d.y *= ( s.y = ( d.y < 0 ) ? -1.f : 1.f );
            int rowOffset = pageOffset + j*rowSize;
            for ( int k = minIndex.z; k <= maxIndex.z; ++k ) {
                d.z = k - gridIndex.z;
                d.z *= ( s.z = ( d.z < 0 ) ? -1.f : 1.f );
                const ParticleGridNode &node = nodes[rowOffset+k];
                float w;
                vec3 wg;
                weightAndGradient( s, d, w, wg );
                velocityGradient += mat3::outerProduct( node.velocity, wg );
                // Particle velocities
                v_PIC += node.velocity * w;
                dv_FLIP += node.velocityChange * w;
            }
        }
    }
    particle.velocity = (1.f-alpha)*v_PIC + alpha*(particle.velocity+dv_FLIP);
}

//__host__ __device__ __forceinline__
//bool withinBoundsInclusive( const float &v, const float &min, const float &max ) { return (v >= min && v <= max); }

//__host__ __device__ __forceinline__
//bool withinBoundsInclusive( const glm::ivec3 &v, const glm::ivec3 &min, const glm::ivec3 &max ) { return  withinBoundsInclusive(v.x, min.x, max.x)
//                                                                                                            && withinBoundsInclusive(v.y, min.y, max.y)
//                                                                                                          && withinBoundsInclusive(v.z, min.z, max.z);}

//__device__ void atomicAdd(vec3 *add, vec3 toAdd){
//    atomicAdd(&(add->x), toAdd.x);
//    atomicAdd(&(add->y), toAdd.y);
//    atomicAdd(&(add->z), toAdd.z);
//}

//__device__ void processGridVelocitiesFast( Particle *particles, const Grid &grid, const ParticleGridNode *nodes, mat3 &velocityGradient, float alpha )
//{

//    int tid = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.y*blockDim.x;

//    Particle &particle = particles[tid];

//    glm::ivec3 ijk;
//    gridIndexToIJK( threadIdx.y, glm::ivec3(4,4,4), ijk );

//    vec3 gridPos = (particle.position-grid.pos)/grid.h;
//    ijk.x += (int) gridPos.x-1;
//    ijk.y += (int) gridPos.y-1;
//    ijk.z += (int) gridPos.z-1;

//    if ( withinBoundsInclusive(ijk,glm::ivec3(0,0,0),grid.dim) ) {

//        ParticleGridNode &node = nodes[getGridIndex(ijk, grid.dim+1)];
//        float w;
//        vec3 wg;
//        vec3 nodePos( ijk.x, ijk.y, ijk.z );
//        weightAndGradient( gridPos-nodePos, w, wg );



//    }


//}

__device__ void updateParticleDeformationGradients( Particle &particle, const mat3 &velocityGradient, float timeStep,
                                                    float criticalCompression, float criticalStretch )
{
    // Temporarily assign all deformation to elastic portion
//    mat3 F = (mat3(1.f) + timeStep*velocityGradient) * particle.elasticF;
    mat3 F = mat3::addIdentity(timeStep*velocityGradient) * particle.elasticF;

    // Clamp the singular values
    mat3 W, S, Sinv, V;
    computeSVD( F, W, S, V );

    S = mat3( CLAMP( S[0], criticalCompression, criticalStretch ), 0.f, 0.f,
              0.f, CLAMP( S[4], criticalCompression, criticalStretch ), 0.f,
              0.f, 0.f, CLAMP( S[8], criticalCompression, criticalStretch ) );
    Sinv = mat3( 1.f/S[0], 0.f, 0.f,
                 0.f, 1.f/S[4], 0.f,
                 0.f, 0.f, 1.f/S[8] );

    // Compute final deformation components
    particle.elasticF = mat3::multiplyADBt( W, S, V );
    particle.plasticF = mat3::multiplyADBt( V, Sinv, W ) * particle.plasticF;
}

__device__ void checkForCollisions( Particle &particle )
{



}

// NOTE: assumes particleCount % blockDim.x = 0, so tid is never out of range!
// criticalCompression = 1 - theta_c
// criticalStretch = 1 + theta_s
__global__ void updateParticlesFromGrid( Particle *particles, const ParticleGrid grid, const ParticleGridNode *nodes, float timeStep,
                                         float criticalCompression, float criticalStretch, float alpha )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    Particle &particle = particles[tid];

    // Update particle velocities and fill in velocity gradient for deformation gradient computation
    mat3 velocityGradient = mat3( 0.f );
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

__host__ __device__ void print( const mat3 &M )
{
    for ( int i = 0; i < 3; ++i ) {
        for ( int j = 0; j < 3; j++ ) {
            printf( "    %10.5f", M[3*j+i] );
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

#define TESTTHIS( B ) if ( !(B) ) { printf( "FAILED!\n"); failed++; } else { printf( "PASSED.\n" ); passed++; }

__global__ void matrixTestKernel()
{
    printf( "Matrix tests:\n" );

    int failed = 0, passed = 0;

    mat3 I(1.f);

    mat3 A( 1, 2, 3,
            4, 5, 6,
            7, 8, 9 );

    printf( "    Testing GLM conversion: " );
    TESTTHIS( mat3::equals(A, mat3(A.toGLM())) );

    printf( "    Testing identity multiplication: " );
    TESTTHIS( mat3::equals(A, I*A) && mat3::equals(A, A*I) );

    mat3 B( 2, 5, 8,
            9, 3, 5,
            4, 8, 9 );

    mat3 AB( 78,  93, 108,
             56,  73,  90,
             99, 120, 141 );

    mat3 BA( 32, 35, 45,
             77, 83, 111,
             122, 131, 177 );

    printf( "    Testing multiplication: " );
    TESTTHIS( mat3::equals(A*B, AB) && mat3::equals(B*A, BA) );

    printf( "    Testing addition: " );
    mat3 ApB( 3, 7, 11,
              13, 8, 11,
              11, 16, 18 );

    TESTTHIS( mat3::equals(ApB, A+B) && mat3::equals(ApB, B+A) );

    printf( "    Testing subtraction: " );
    mat3 AmB( -1, -5, 3,
              -3, 2, 0,
              -5, 1, 0 );
    mat3 BmA( 1, 5, -3,
              3, -2, 0,
              5, -1, 0 );
    TESTTHIS( mat3::equals(AmB, A-B) && mat3::equals(BmA, B-A) );

    printf( "    Testing scalar multiplication: " );
    mat3 As( 2, 4, 6,
             8, 10, 12,
             14, 16, 18 );
    TESTTHIS( mat3::equals(As, A*2.f) && mat3::equals(As, 2.f*A) );

    printf( "    Testing scalar division: " );
    mat3 Ad( 0.5f, 1.f, 1.5f,
             2.f, 2.5f, 3.f,
             3.5f, 4.f, 4.5f );
    TESTTHIS( mat3::equals(Ad, A/2.f) );

    printf( "    Testing transpose: " );
    mat3 At( 1, 4, 7,
             2, 5, 8,
             3, 6, 9 );
    TESTTHIS( mat3::equals(At, mat3::transpose(A)) );

    printf( "    Testing multiplyTransposeL: " );
    TESTTHIS( mat3::equals(mat3::multiplyAtB(A, B), mat3::transpose(A)*B) );

    printf( "    Testing outer product: " );
    vec3 v( 10, 5, 1 ), w( 8, 2, 9 );
    mat3 M = mat3::outerProduct( v, w );
    mat3 expected = mat3::transpose(mat3(80, 20, 90, 40, 10, 45, 8, 2, 9));
    TESTTHIS( mat3::equals(expected, M) );

    printf( "    Testing determinant: " );
    TESTTHIS( mat3::determinant(mat3(4,0,2,0,2,0,1,0,1)) == 4.f );

    printf( "    FAILED %d TESTS (OUT OF %d)\n", failed, failed+passed );

}

__global__ void velocityTest( Particle *particles, glm::mat3 *gradients, const Grid grid, ParticleGridNode *nodes, float alpha )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Particle &particle = particles[tid];
    mat3 velocityGradient = gradients[tid];
    velocityGradient = glm::mat3(0.f);
    processGridVelocities( particle, grid, nodes, velocityGradient, alpha );
    gradients[tid] = velocityGradient.toGLM();
}

__host__ void valueTests()
{
    const int dim = 8;
    ParticleGrid grid;
    grid.dim = glm::ivec3( dim, dim, dim );
    grid.h = 1.f/dim;
    grid.pos = vec3(0,0,0);

    int nParticles = dim*dim*dim;
    printf( "    Generating %d particles (%.2f MB)...\n",
            nParticles, nParticles*sizeof(Particle)/1e6 );
    fflush(stdout);
    Particle *particles = new Particle[nParticles];
    for ( int i = 0; i < dim; ++i ) {
        for ( int j = 0; j < dim; ++j ) {
            for ( int k = 0; k < dim; ++k ) {
                Particle particle;
                particle.position = grid.pos + grid.h*vec3( i+0.5f, j+0.5f, k+0.5f );
                particle.velocity = vec3( 0.f, -0.124f, 0.f );
                particle.elasticF = mat3(1.f);
                particle.plasticF = mat3(1.f);
                particles[i*dim*dim+j*dim+k] = particle;
            }
        }
    }

    printf( "    Generating %d grid nodes (%.2f MB)...\n",
            (dim+1)*(dim+1)*(dim+1), (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGridNode)/1e6 );
    fflush(stdout);
    ParticleGridNode *nodes = grid.createNodes();
    for ( int i = 0; i <= dim; ++i ) {
        for ( int j = 0; j <= dim; ++j ) {
            for ( int k = 0; k <= dim; ++k ) {
                ParticleGridNode node;
                node.velocity = vec3( 0.f, -0.125f, 0.f );
                node.velocityChange = vec3( 0.f, -0.001f, 0.f );
                nodes[i*(dim+1)*(dim+1)+j*(dim+1)+k] = node;
            }
        }
    }


    printf( "    Allocating kernel resources...\n" ); fflush(stdout);
    Particle *devParticles;
    ParticleGridNode *devNodes;
    glm::mat3 *devGradients;
    cudaError_t error;
    error = cudaMalloc( &devParticles, nParticles*sizeof(Particle) );
    error = cudaMemcpy( devParticles, particles, nParticles*sizeof(Particle), cudaMemcpyHostToDevice );
    error = cudaMalloc( &devNodes, (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGridNode) );
    error = cudaMemcpy( devNodes, nodes, (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGridNode), cudaMemcpyHostToDevice );
    error = cudaMalloc( &devGradients, nParticles*sizeof(glm::mat3) );

    static const int blockSize = 128;
    if ( error != cudaSuccess ) {
        printf( "    FAILED: %s\n", _cudaGetErrorEnum(error) );
    } else {
        for ( int i = 0; i < 5; ++i ) {
            TIME( "    Launching velocity kernel... ", "finished.\n",
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

    for ( int i = 0; i < dim; ++i ) {
        for ( int j = 0; j < dim; ++j ) {
            for ( int k = 0; k < dim; ++k ) {
                Particle particle;
                particle.position = grid.pos + grid.h*vec3( i+0.5f, j+0.5f, k+0.5f );
                particle.velocity = vec3( 0.f, -0.124, 0.f );
                particle.elasticF = mat3(1.f);
                particle.plasticF = mat3(1.f);
                particles[i*dim*dim+j*dim+k] = particle;
            }
        }
    }
    cudaMemcpy( devParticles, particles, nParticles*sizeof(Particle), cudaMemcpyHostToDevice );

    TIME( "    Launching full kernel... ", "finished\n",
          updateParticlesFromGrid<<< (nParticles+blockSize-1)/blockSize, blockSize >>>( devParticles, grid, devNodes, 1.0f, 0.8f, 1.2f, 0.95f );
          error = cudaDeviceSynchronize();
    );

    cudaMemcpy( particles, devParticles, nParticles*sizeof(Particle), cudaMemcpyDeviceToHost );

    mat3 elastic = mat3::transpose( mat3( 1.000000000000000, 0.000000000000000,  0.000000000000000,
                                          -0.014980740017361, 0.985019259982640, -0.014980740017361,
                                          -0.000000000000000, 0.000000000000000,  1.000000000000000 ) );
    mat3 plastic = mat3::transpose( mat3( 1.000000000000000, -0.000000000000000, -0.000000000000000,
                                          0.015208575736504,  1.015208575736504,  0.015208575736504,
                                          0.000000000000000,  0.000000000000000,  1.000000000000000 ) );
    if ( !mat3::equals(particles[511].elasticF, elastic ) ) {
        printf( "    FAILED: expected elastic deformation gradient:\n" );
        print( elastic );
        printf( "            got:\n" );
        print( particles[511].elasticF );
    } else if ( !mat3::equals(particles[511].plasticF, plastic ) ) {
        printf( "    FAILED: expected plastic deformation gradient:\n" );
        print( plastic );
        printf( "            got:\n" );
        print( particles[511].plasticF );
    } else {
        printf( "    ALL DEFORMATION GRADIENT CHECKS PASSED.\n" );
    }

    printf( "    Freeing kernel resources...\n" ); fflush(stdout);
    cudaFree( devParticles );
    cudaFree( devNodes );
    cudaFree( devGradients );
    delete [] particles;
    delete [] nodes;
    delete [] gradients;
}

__host__ void timeTests()
{
    const int dim = 50;
    ParticleGrid grid;
    grid.dim = glm::ivec3( dim, dim, dim );
    grid.h = 1.f/dim;
    grid.pos = vec3(0,0,0);

    int nParticles = 5000*32;
    printf( "    Generating %d particles (%.2f MB)...\n",
            nParticles, nParticles*sizeof(Particle)/1e6 );
    fflush(stdout);
    Particle *particles = new Particle[nParticles];
    for ( int i = 0; i < nParticles; ++i ) {
        Particle particle;
        particle.position = grid.pos + vec3( urand(), urand(), urand() );
        particle.velocity = vec3( 0.f, -0.124f, 0.f );
        particle.elasticF = mat3(1.f);
        particle.plasticF = mat3(1.f);
        particles[i] = particle;
    }

    printf( "    Generating %d grid nodes (%.2f MB)...\n",
            (dim+1)*(dim+1)*(dim+1), (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGridNode)/1e6 );
    fflush(stdout);
    ParticleGridNode *nodes = grid.createNodes();
    for ( int i = 0; i <= dim; ++i ) {
        for ( int j = 0; j <= dim; ++j ) {
            for ( int k = 0; k <= dim; ++k ) {
                ParticleGridNode node;
                node.velocity = vec3( 0.f, -0.125f, 0.f );
                node.velocityChange = vec3( 0.f, -0.001f, 0.f );
                nodes[i*(dim+1)*(dim+1)+j*(dim+1)+k] = node;
            }
        }
    }

    printf( "    Allocating kernel resources...\n" ); fflush(stdout);
    Particle *devParticles;
    ParticleGridNode *devNodes;
    cudaError_t error;
    error = cudaMalloc( &devParticles, nParticles*sizeof(Particle) );
    error = cudaMalloc( &devNodes, (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGridNode) );

//    static const int blockSizes[] = { 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512 };
    static const int nBlocks = 16;

    for ( int i = 0; i < nBlocks; ++i ) {
        error = cudaMemcpy( devParticles, particles, nParticles*sizeof(Particle), cudaMemcpyHostToDevice );
        error = cudaMemcpy( devNodes, nodes, (dim+1)*(dim+1)*(dim+1)*sizeof(ParticleGridNode), cudaMemcpyHostToDevice );
//        int blockSize = blockSizes[i];
        int blockSize = 256;
        printf( "    Block size = %3d; ", blockSize ); fflush(stdout);
        TIME( " Launching full kernel... ", "finished\n",
            updateParticlesFromGrid<<< (nParticles+blockSize-1)/blockSize, blockSize >>>( devParticles, grid, devNodes, 1e-5, 0.8f, 1.2f, 0.95f );
            error = cudaDeviceSynchronize();
        );
        if ( error != cudaSuccess ) break;
    }

    if ( error != cudaSuccess ) {
        printf( "    FAILED: %s\n", _cudaGetErrorEnum(error) );
    } else {
        printf( "    PASSED.\n" );
    }

    printf( "    Freeing kernel resources...\n" ); fflush(stdout);
    cudaFree( devParticles );
    cudaFree( devNodes );
    delete [] particles;
    delete [] nodes;
}

__host__ void grid2ParticlesTests()
{
    matrixTestKernel<<<1,1>>>();
    cudaDeviceSynchronize();

    printf( "Grid -> Particles Tests:\n" );

    valueTests();
    timeTests();

    printf( "Done.\n" );



}

#endif // MAX_CU
