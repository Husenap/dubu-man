#include "lambertian.cuh"

namespace dubu_man{
    __device__ bool lambertian::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
                                        curandState &rand_state) const {
        auto scatter_direction = rec.normal + vec3::random_unit_vector(rand_state);
        if (near_zero(scatter_direction)) {
            scatter_direction = rec.normal;
        }

        scattered = ray(rec.p, scatter_direction);
        attenuation = m_albedo;

        return true;
    }

    __device__ color lambertian::get_albedo(const hit_record &rec) const {
        return m_albedo;
    }
}