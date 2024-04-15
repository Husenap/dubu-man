#pragma once

#include "material.cuh"

namespace dubu_man {
    class metal : public material {
        color m_albedo;
        float m_fuzz;
    public:
        __device__ metal(color const &albedo, float fuzz) : m_albedo(albedo), m_fuzz(fuzz) {}

        __device__ bool scatter(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                                curandState &rand_state) const override;

        __device__ color get_albedo(hit_record const &rec) const override;
    };
} // dubu_man