#include <cuda.h>
#include <stdio.h>
#include "CUDA_helpers.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher
#endif


extern "C"
void runCudaPart();

#define N   10

__global__ void add(int *a, int *b, int *c)
{
    // tid is which GPU block we are operating on
    // can be 1D, 2D, or even 3D indexed for image/grid-type operations
    int tid=blockIdx.x;
    if (tid<N) // safeguard in case GPU does something weird, but not really necessary
        c[tid] = a[tid] + b[tid];
}

void runCudaPart()
{
    int a[N], b[N], c[N];   // CPU-side data
    int *dev_a, *dev_b, *dev_c; // GPU-side pointers

    // allocate memory to GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_a,N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b,N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c,N*sizeof(int)));

    // fill arrays a,b on CPU. we could do this on GPU too if we wanted
    for (int i=0; i<N; ++i)
    {
        a[i]=-i;
        b[i]=i*i;
    }
    // copy arrays a,b to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a,a,N*sizeof(int),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b,b,N*sizeof(int),cudaMemcpyHostToDevice));
    add<<<N,1>>>(dev_a,dev_b,dev_c);
    // copy array c back from GPU to CPU
    HANDLE_ERROR(cudaMemcpy(c,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost));
    // display the results
    for (int i=0;i<N;++i)
    {
        printf("%d + %d = %d\n",a[i],b[i],c[i]);
    }
    // free memory on GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
