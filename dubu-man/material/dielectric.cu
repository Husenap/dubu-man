#include "dielectric.cuh"

namespace dubu_man {
    __device__ float dielectric::reflectance(const float cosine, const float ior) {
        auto r0 = (1 - ior) / (1 + ior);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1.0f - cosine), 5.0f);
    }

    __device__ bool dielectric::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
                                        curandState &rand_state) const {
        attenuation = color{1.0f};

        const auto ior = rec.front_face ? (1.0f / m_ior) : m_ior;
        const auto unit_direction = normalize(r_in.direction());
        const auto cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
        const auto sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = ior * sin_theta > 1.0f;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ior) > curand_uniform(&rand_state)) {
            direction = reflect(unit_direction, rec.normal);
        } else {

            direction = refract(unit_direction, rec.normal, ior);
        }

        scattered = ray(rec.p, direction);

        return true;
    }

    __device__ color dielectric::get_albedo(const hit_record &rec) const {
        return color{1.0f};
    }
}
