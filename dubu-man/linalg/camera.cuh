#pragma once

#include <curand_kernel.h>

#include "ray.cuh"
#include "vec3.cuh"

namespace dubu_man {
class camera {
  vec3  m_pixel_delta_u{};
  vec3  m_pixel_delta_v{};
  vec3  m_pixel00_loc{};
  vec3  m_center{};
  float m_defocus_angle{};
  vec3  m_defocus_disk_u{};
  vec3  m_defocus_disk_v{};

public:
  struct camera_settings {
    size_t image_width{400};
    float  aspect_ratio{16.0f / 9.0f};
    size_t max_bounces{10};

    float  vfov{90.0f};
    point3 look_from{};
    point3 look_at{0, 0, -1};
    vec3   vup{0, 1, 0};

    float defocus_angle{};
    float focus_dist{10.0f};
  };

  size_t          image_width{};
  size_t          image_height{};
  size_t          max_bounces{};
  camera_settings settings{};

  camera() = default;
  __host__ __device__ explicit camera(const camera_settings& settings);

  __device__ ray get_ray(size_t i, size_t j, curandState& rand_state) const;

  static __device__ vec3 sample_square(curandState& rand_state);

  __device__ vec3 sample_defocus_disk(curandState& rand_state) const;
};

} // namespace dubu_man