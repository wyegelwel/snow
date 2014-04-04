#include <cuda.h>
#include <glm/glm.hpp>
#include <stdio.h>


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher
#endif

extern "C"
void runCudaPart();


void runCudaPart()
{
    // all your cuda code here
    printf("Hello, world from the device!");
    glm::vec4 v(0.0f);
}
