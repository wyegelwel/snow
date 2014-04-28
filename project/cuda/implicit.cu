/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   implicit.cu
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 26 Apr 2014
**
**************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_INCLUDE
#include "sim/particle.h"

/**
 * Called over particles over nodes the particle affects. (numParticles * 64)
 *
 * Recommended:
 *  dim3 blockDim = dim3(numParticles / threadCount, 64);
 *  dim3 threadDim = dim3(threadCount/64, 64);
 *
 **/
__global__ void computedF(Particle *particles, Grid *grid, vec3 *du, mat3 *dFs){
    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    Particle &particle = particles[particleIdx];
    mat3 &dF = dFs[particleIdx];

    vec3 particleGridPos = (particle.position - grid->pos)/grid->h;
    glm::ivec3 currIJK;
    gridIndexToIJK(threadIdx.y, glm::ivec3(4,4,4), currIJK);
    currIJK.x += (int) particleGridPos.x - 1; currIJK.y += (int) particleGridPos.y - 1; currIJK.z += (int) particleGridPos.z - 1;

    if (withinBoundsInclusive(currIJK, glm::ivec3(0,0,0), grid->dim)){
        vec3 du_j = du[getGridIndex(currIJK, grid->dim+1)];

        float w;
        vec3 wg;
        vec3 nodePosition(currIJK.x, currIJK.y, currIJK.z);
        weightAndGradient(particleGridPos-nodePosition, w, wg);

        atomicAdd(&dF, mat3::outerProduct(du_j, wg) * particle.elasticF);
     }

}

//__global__ void computeAp(Particle *particles, Grid *grid, vec3 *du, mat3 *dFs, mat3 *Aps){
//    int particleIdx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

//    Particle &particle = particles[particleIdx];
//    mat3 &dF = dFs[particleIdx];
//    mat3 &Ap = Aps[particleIdx];

//    mat3 &Fp = particle.plasticF; //for the sake of making the code look like the math
//    mat3 &Fe = particle.elasticF;

//    float Jpp = mat3::determinant(Fp);
//    float Jep = mat3::determinant(Fe);

//    mat3 Re;
//    computePD(Fe, Re);

//    float muFp = material->mu*__expf(material->xi*(1-Jpp));
//    float lambdaFp = material->lambda*__expf(material->xi*(1-Jpp));

//    mat3 dRe = Re; // Need to actually compute dRe

//    mat3 jFe_invTrans = Jep*mat3::transpose(mat3::inverse(Fe));

//    Ap = (2*muFp*(dF - dRe) +lambdaFp*jFe_invTrans*mat3::innerProduct(jFe_invTrans, dF) + lambdaFp*(Jep - 1));

////    sigma = (2*muFp*(Fe-Re)*mat3::transpose(Fe)+lambdaFp*(Jep-1)*Jep*mat3(1.0f)) * (particle.volume);
////    sigma = (2*muFp*mat3::multiplyABt(Fe-Re, Fe) + mat3(lambdaFp*(Jep-1)*Jep)) * -particle.volume;
//}
