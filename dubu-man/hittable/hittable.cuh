#pragma once

#include "hit_record.cuh"
#include "../material/material.cuh"
#include "../linalg/ray.cuh"
#include "../linalg/interval.cuh"

namespace dubu_man {
    class hittable {
    public:
        __device__ virtual ~hittable() {}

        __device__ virtual bool hit(ray const &r, interval ray_t, hit_record &rec) const = 0;
    };
} // dubu_man