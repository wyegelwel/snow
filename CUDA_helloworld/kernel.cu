#include <cuda.h>
#include <glm/glm.hpp>
#include <stdio.h>


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher
#endif

extern "C"
void runCudaPart();

__global__ void helloCUDA(glm::vec3 v)
{
    int tid = blockIdx.x;
    printf("Hello block %d thread %d, x=%f\n",tid , threadIdx.x, v.x);
}

void runCudaPart()
{
    // all your cuda code here
    glm::vec3 v(0.1f, 0.2f, 0.3f);
//    helloCUDA<<<1, 5>>>(v); // 1 block, 5 GPU threads
    helloCUDA<<<5,1>>>(v); // 5 blocks, 1 GPU thread each
    cudaDeviceSynchronize();
}
