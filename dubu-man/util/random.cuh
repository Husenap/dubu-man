#pragma once

#include <curand_kernel.h>

__device__ inline float random01(curandState& rand_state) {
    // Returns a random real in [0,1).
    return curand_uniform(&rand_state);
}

__device__ inline float random_range(float min, float max, curandState& rand_state) {
    // Returns a random real in [min,max).
    return min + (max-min)*random01(rand_state);
}
