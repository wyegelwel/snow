/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   tim.cu
**   Author: taparson
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef TIM_CU
#define TIM_CU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <glm/geometric.hpp>

#define CUDA_INCLUDE
#include "sim/particle.h"
#include "cuda/functions.h"

extern "C"
void cumulativeSumTests();
void CSTest1();
void CSTest2();
void CSTest3();
void CSTest4();
void CSTest5();

__global__ void cumulativeSum(int *array, int M)  {
    int sum = 0;
    for(int i = 0; i < M; i++)  {
        sum+=array[i];
        array[i] = sum;
    }
}

void cumulativeSumTests()
{
    printf("running cumulative sum tests...\n");
    CSTest1();
    CSTest2();
    CSTest3();
    CSTest4();
    CSTest5();
    printf("done running cumulative sum tests\n");
}

void CSTest1()  {
    int array[5] = {0,1,2,3,4};
    int expected[5] = {0,1,3,6,10};
    printf("running test on array: [%d,%d,%d,%d,%d]...\n",array[0],array[1],array[2],array[3],array[4]);
    int *dev_array;
    checkCudaErrors(cudaMalloc((void**) &dev_array, 5*sizeof(int)));
    checkCudaErrors(cudaMemcpy(dev_array,array,5*sizeof(int),cudaMemcpyHostToDevice));
    cumulativeSum<<<1,1>>>(dev_array,5);
    cudaMemcpy(array,dev_array,5*sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dev_array);
    for (int i = 0; i < 5; i++)  {
        if (array[i] != expected[i])  {
            printf("failed test %d",1);
            printf("expected array: {%d,%d,%d,%d,%d}",expected[0],expected[1],expected[2],expected[3],expected[4]);
            printf("    got: {%d,%d,%d,%d,%d}\n",array[0],array[1],array[2],array[3],array[4]);
            break;
        }
    }
}

void CSTest2()  {
    int array[5] = {5,1,2,3,4};
    int expected[5] = {5,6,8,11,15};
    printf("running test on array: [%d,%d,%d,%d,%d]...\n",array[0],array[1],array[2],array[3],array[4]);
    int *dev_array;
    checkCudaErrors(cudaMalloc((void**) &dev_array, 5*sizeof(int)));
    checkCudaErrors(cudaMemcpy(dev_array,array,5*sizeof(int),cudaMemcpyHostToDevice));
    cumulativeSum<<<1,1>>>(dev_array,5);
    cudaMemcpy(array,dev_array,5*sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dev_array);
    for (int i = 0; i < 5; i++)  {
        if (array[i] != expected[i])  {
            printf("failed test %d",1);
            printf("expected array: {%d,%d,%d,%d,%d}",expected[0],expected[1],expected[2],expected[3],expected[4]);
            printf("    got: {%d,%d,%d,%d,%d}\n",array[0],array[1],array[2],array[3],array[4]);
            break;
        }
    }
}

void CSTest3()  {
    int array[1] = {5};
    int expected[1] = {5};
    printf("running test on array: [%d]...\n",array[0]);
    int *dev_array;
    checkCudaErrors(cudaMalloc((void**) &dev_array, 1*sizeof(int)));
    checkCudaErrors(cudaMemcpy(dev_array,array,1*sizeof(int),cudaMemcpyHostToDevice));
    cumulativeSum<<<1,1>>>(dev_array,1);
    cudaMemcpy(array,dev_array,1*sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dev_array);
    for (int i = 0; i < 1; i++)  {
        if (array[i] != expected[i])  {
            printf("failed test %d",1);
            printf("expected array: {%d}",expected[0]);
            printf("    got: {%d}\n",array[0]);
            break;
        }
    }
}

void CSTest4()  {
    int array[1] = {0};
    int expected[1] = {0};
    printf("running test on array: [%d]...\n",array[0]);
    int *dev_array;
    checkCudaErrors(cudaMalloc((void**) &dev_array, 1*sizeof(int)));
    checkCudaErrors(cudaMemcpy(dev_array,array,1*sizeof(int),cudaMemcpyHostToDevice));
    cumulativeSum<<<1,1>>>(dev_array,1);
    cudaMemcpy(array,dev_array,1*sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dev_array);
    for (int i = 0; i < 1; i++)  {
        if (array[i] != expected[i])  {
            printf("failed test %d",1);
            printf("expected array: {%d}",expected[0]);
            printf("    got: {%d}\n",array[0]);
            break;
        }
    }
}

void CSTest5()  {
    int array[0] = {};
    int expected[0] = {};
    printf("running test on array: []...\n");
    int *dev_array;
    checkCudaErrors(cudaMalloc((void**) &dev_array, 0*sizeof(int)));
    checkCudaErrors(cudaMemcpy(dev_array,array,0*sizeof(int),cudaMemcpyHostToDevice));
    cumulativeSum<<<1,1>>>(dev_array,0);
    cudaMemcpy(array,dev_array,0*sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dev_array);
    for (int i = 0; i < 0; i++)  {
        if (array[i] != expected[i])  {
            printf("failed test %d",1);
            printf("expected array: {}",expected[0]);
            printf("    got: {}\n");
            break;
        }
    }
}

#endif // TIM_CU

