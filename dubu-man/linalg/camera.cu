#include "camera.cuh"

#include "trig.cuh"

namespace dubu_man {
    __device__ __host__ camera::camera(const camera_settings &settings) {
        image_width = settings.image_width;
        image_height = static_cast<size_t>(fmax(
                static_cast<float>(image_width) / settings.aspect_ratio, 1.0f));
        samples_per_pixel = settings.samples_per_pixel;
        max_bounces = settings.max_bounces;
        pixel_samples_scale = 1.0f / static_cast<float>(samples_per_pixel);
        m_defocus_angle = settings.defocus_angle;

        m_center = settings.look_from;

        const auto theta = degrees_to_radians(settings.vfov);
        const auto h = tan(theta / 2.0f);
        const auto viewport_height = 2.0f * h * settings.focus_dist;
        const auto viewport_width =
                viewport_height * (static_cast<float>(image_width) / static_cast<float>(image_height));

        const auto w = normalize(settings.look_from - settings.look_at);
        const auto u = normalize(cross(settings.vup, w));
        const auto v = cross(w, u);

        const auto viewport_u = viewport_width * u;
        const auto viewport_v = -viewport_height * v;

        m_pixel_delta_u = viewport_u / static_cast<float>(image_width);
        m_pixel_delta_v = viewport_v / static_cast<float>(image_height);

        const auto viewport_upper_left = m_center
                                         - settings.focus_dist * w
                                         - viewport_u * 0.5f
                                         - viewport_v * 0.5f;
        m_pixel00_loc = viewport_upper_left + 0.5 * (m_pixel_delta_u + m_pixel_delta_v);

        const auto defocus_radius = settings.focus_dist * tan(degrees_to_radians(m_defocus_angle / 2.0f));
        m_defocus_disk_u = u * defocus_radius;
        m_defocus_disk_v = v * defocus_radius;
    }

    __device__ ray camera::get_ray(size_t i, size_t j, curandState &rand_state) const {
        const auto offset = sample_square(rand_state);
        const auto pixel_sample = m_pixel00_loc
                                  + (offset.x + static_cast<float>(i)) * m_pixel_delta_u
                                  + (offset.y + static_cast<float>(j)) * m_pixel_delta_v;

        const auto ray_origin = m_defocus_angle <= 0.0f ? m_center : sample_defocus_disk(rand_state);
        const auto ray_direction = pixel_sample - ray_origin;

        return {ray_origin, ray_direction};
    }

    __device__ vec3 camera::sample_square(curandState &rand_state) {
        const auto px = -0.5f + curand_uniform(&rand_state);
        const auto py = -0.5f + curand_uniform(&rand_state);
        return {px, py, 0.0f};
    }

    __device__ vec3 camera::sample_defocus_disk(curandState &rand_state) const {
        const auto p = vec3::random_in_unit_disk(rand_state);
        return m_center + p.x * m_defocus_disk_u + p.y * m_defocus_disk_v;
    }
}