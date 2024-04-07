#pragma once

#include <curand_kernel.h>
#include "../settings.cuh"
#include "vec3.cuh"
#include "ray.cuh"

namespace dubu_man {
    class camera {
        point3 m_center{};
        vec3 m_pixel_delta_u{};
        vec3 m_pixel_delta_v{};
        vec3 m_pixel00_loc{};

    public:
        __host__ explicit camera(point3 center);

        __device__ ray get_ray(size_t i, size_t j) const;
        __device__ ray get_sub_pixel_ray(size_t i, size_t j, curandState& rand_state) const;
        __device__ vec3 pixel_sample_square(curandState& rand_state) const;
    };

}