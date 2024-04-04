#pragma once

#include "float3_extension.cuh"

namespace dubu_man {
    class ray {
    public:
        ray() = default;

        __device__ ray(point3 const origin, vec3 const direction) : m_origin(origin), m_direction(direction) {}

        __device__  point3 origin() const { return m_origin; }

        __device__  vec3 direction() const { return m_direction; }

        __device__ point3 at(component t) const {
            return m_origin + t * m_direction;
        }

    private:
        point3 m_origin;
        vec3 m_direction;
    };
} // dubu_man
