#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "hittable/hittable.cuh"
#include "linalg/camera.cuh"
#include "linalg/vec3.cuh"

namespace dubu_man {
struct PixelData {
  vec3 color;
  vec3 albedo;
  vec3 normal;
};

__global__ void render_init(const camera* cam, curandState* rand_state);

__host__ __device__ float linear_to_srgb(float value);

__device__ color ray_color(ray const& r, const camera* cam, hittable2 world, curandState& rand_state);

__global__ void render(PixelData framebuffer[], size_t framebuffer_pitch, const camera* cam, hittable2* world, curandState* rand_state);

} // namespace dubu_man