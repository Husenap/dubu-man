#pragma once

#include "../linalg/vec3.cuh"
#include "../linalg/ray.cuh"
#include "../material/material.cuh"

namespace dubu_man {
    struct hit_record {
        point3 p;
        vec3 normal;
        material2 material{};
        float t{};
        bool front_face{};

        __device__ void set_face_normal(ray const &r, vec3 const &outward_normal) {
            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
    };
}