#pragma once

#include "hit_record.cuh"
#include "../material/material.cuh"
#include "../linalg/ray.cuh"

namespace dubu_man {
    class hittable {
    public:
        __device__ virtual ~hittable() {
            printf("~hittable(), ");
        };

        __device__ virtual bool hit(ray const &r, float ray_tmin, float ray_tmax, hit_record &rec) const = 0;
    };
} // dubu_man