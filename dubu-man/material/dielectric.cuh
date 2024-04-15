#pragma once

#include "material.cuh"

namespace dubu_man {
    class dielectric : public material {
        float m_ior;

        static __device__ float reflectance(const float cosine, const float ior);

    public:
        __device__ dielectric(float index_of_refraction) : m_ior(index_of_refraction) {}

        __device__ bool scatter(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                                curandState &rand_state) const override;

        __device__ color get_albedo(hit_record const &rec) const override;
    };
} // dubu_man