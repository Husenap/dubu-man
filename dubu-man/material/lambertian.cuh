#pragma once

#include "material.cuh"

namespace dubu_man {
    class lambertian : public material {
        color m_albedo;
    public:
        __device__ lambertian(color const &albedo) : m_albedo(albedo) {}

        __device__ bool scatter(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                                curandState &rand_state) const override;

        __device__ color get_albedo(hit_record const &rec) const override;
    };
} // dubu_man