#ifndef DUBU_MAN_CAMERA_CUH
#define DUBU_MAN_CAMERA_CUH

#include "settings.cuh"
#include "float3_extension.cuh"
#include "ray.cuh"

namespace dubu_man {
    class camera {
    public:
        explicit camera(point3 center) {
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

        __device__  ray get_ray(size_t i, size_t j) const {
            const auto pixel_center = m_pixel00_loc
                                      + static_cast<component>(i) * m_pixel_delta_u
                                      + static_cast<component>(j) * m_pixel_delta_v;
            const auto ray_direction = pixel_center - m_center;
            return {m_center, ray_direction};
        }

        point3 m_center{};
        vec3 m_pixel_delta_u{};
        vec3 m_pixel_delta_v{};
        vec3 m_pixel00_loc{};
    };
}


#endif