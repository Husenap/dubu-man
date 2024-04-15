#pragma once

#include "../hittable/hittable.cuh"

namespace dubu_man {
    class material {
    public:
        __device__ virtual ~material() {}

        __device__ virtual bool
        scatter(ray const &r_in, hit_record const &rec, color &attenuation, ray &scattered,
                curandState &rand_state) const = 0;

        __device__ virtual color
        get_albedo(hit_record const &rec) const = 0;
    };
} // dubu_man