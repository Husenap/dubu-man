#pragma once

#include "../linalg/ray.cuh"

namespace dubu_man {
    struct hit_record;

    class material {
    public:
        __device__ virtual ~material() {}

        __device__ virtual bool
        scatter(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                curandState &rand_state) const = 0;

        __device__ virtual color
        get_albedo(hit_record const &rec) const = 0;
    };

    struct material2 {
        enum class material_type {
            lambertian,
            metal,
            dielectric,
        };

        material_type type;
        union {
            struct lambertian_data {
                color albedo;
            } lambertian;
            struct metal_data {
                color albedo;
                float fuzz{};
            } metal;
            struct dielectric_data {
                float ior{};
            } dielectric;
        };

        static __host__ __device__ material2 make_lambertian(color const &albedo);

        static __host__ __device__ material2 make_metal(color const &albedo, float fuzz);

        static __host__ __device__ material2 make_dielectric(float ior);


        __device__ bool scatter(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                                curandState &rand_state) const;

        __device__  color const &get_albedo(hit_record const &rec) const;

    private:
        __device__ bool scatter_lambertian(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                                           curandState &rand_state) const;

        __device__ bool scatter_metal(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                                      curandState &rand_state) const;

        static __device__ float reflectance(const float cosine, const float ior) {
            auto r0 = (1 - ior) / (1 + ior);
            r0 = r0 * r0;
            return r0 + (1 - r0) * pow((1.0f - cosine), 5.0f);
        }

        __device__ bool scatter_dielectric(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                                           curandState &rand_state) const;
    };
} // dubu_man
