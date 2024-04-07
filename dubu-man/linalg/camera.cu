#include "camera.cuh"

namespace dubu_man {
    __host__ camera::camera(point3 center) {
        m_center = center;
        constexpr auto viewport_u = vec3{VIEWPORT_WIDTH, 0, 0};
        constexpr auto viewport_v = vec3{0, -VIEWPORT_HEIGHT, 0};

        m_pixel_delta_u = viewport_u / IMAGE_WIDTH;
        m_pixel_delta_v = viewport_v / IMAGE_HEIGHT;

        const auto viewport_upper_left = m_center
                                         - vec3{0, 0, FOCAL_LENGTH}
                                         - viewport_u * 0.5
                                         - viewport_v * 0.5;

        m_pixel00_loc = viewport_upper_left + 0.5 * (m_pixel_delta_u + m_pixel_delta_v);
    }

    __device__ ray camera::get_ray(size_t i, size_t j) const {
        const auto pixel_center = m_pixel00_loc
                                  + static_cast<float>(i) * m_pixel_delta_u
                                  + static_cast<float>(j) * m_pixel_delta_v;
        const auto ray_direction = pixel_center - m_center;
        return {m_center, ray_direction};
    }
    __device__ ray camera::get_sub_pixel_ray(size_t i, size_t j, curandState &rand_state) const {
        const auto pixel_center = m_pixel00_loc
                                  + static_cast<float>(i) * m_pixel_delta_u
                                  + static_cast<float>(j) * m_pixel_delta_v
                                  + pixel_sample_square(rand_state);
        const auto ray_direction = pixel_center - m_center;
        return {m_center, ray_direction};
    }

    __device__ vec3 camera::pixel_sample_square(curandState &rand_state) const {
        const auto px = -0.5f + curand_uniform(&rand_state);
        const auto py = -0.5f + curand_uniform(&rand_state);
        return px * m_pixel_delta_u + py * m_pixel_delta_v;
    }
}