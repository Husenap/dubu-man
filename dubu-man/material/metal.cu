#include "metal.cuh"

namespace dubu_man {
    __device__ bool metal::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
                                   curandState &rand_state) const {
        const auto reflected = reflect(normalize(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + m_fuzz * vec3::random_unit_vector(rand_state));
        attenuation = m_albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

    __device__ color metal::get_albedo(const hit_record &rec) const {
        return m_albedo;
    }
}
