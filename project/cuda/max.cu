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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "glm/common.hpp"
#include "glm/geometric.hpp"

#define CUDA_INCLUDE
#include "common/common.h"
#include "common/math.h"
#include "cuda/functions.h"
#include "geometry/bbox.h"
#include "geometry/mesh.h"
#include "sim/particle.h"
#include "sim/grid.h"

#endif // MAX_CU
