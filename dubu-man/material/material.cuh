#pragma once

#include "../hittable/hittable.cuh"

namespace dubu_man {
    class material {
    public:
        __device__ virtual ~material() {
            printf("~material(), ");
        };

        __device__ virtual bool
        scatter(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                curandState &rand_state) const = 0;

        __device__ virtual color
        get_albedo(hit_record const &rec) const = 0;
    };

    class lambertian : public material {
        color m_albedo;
    public:
        __device__ lambertian(color const &albedo) : m_albedo(albedo) {}

        __device__ ~lambertian() override {
            printf("~lambertian(), ");
        }

        __device__ bool scatter(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                                curandState &rand_state) const override {
            auto scatter_direction = rec.normal + vec3::random_unit_vector(rand_state);
            if (near_zero(scatter_direction)) {
                scatter_direction = rec.normal;
            }

            scattered = ray(rec.p, scatter_direction);
            attenuation = m_albedo;

            return true;
        }

        __device__ color get_albedo(hit_record const &rec) const override {
            return m_albedo;
        }
    };

    class metal : public material {
        color m_albedo;
        float m_fuzz;
    public:
        __device__ metal(color const &albedo, float fuzz) : m_albedo(albedo), m_fuzz(fuzz) {}

        __device__ ~metal() override {
            printf("~metal(), ");
        }

        __device__ bool scatter(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                                curandState &rand_state) const override {
            const auto reflected = reflect(normalize(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + m_fuzz * vec3::random_unit_vector(rand_state));
            attenuation = m_albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

        __device__ color get_albedo(hit_record const &rec) const override {
            return m_albedo;
        }
    };
} // dubu_man