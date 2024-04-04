#ifndef DUBU_MAN_FLOAT3_EXTENSION_CUH
#define DUBU_MAN_FLOAT3_EXTENSION_CUH

#include <iostream>


using component = float;
using vec3 = float3;
using point3 = vec3;
using color = vec3;

__device__ __host__ vec3 operator+(vec3 const &a, vec3 const &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __host__ vec3 operator-(vec3 const &a, vec3 const &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __host__ vec3 operator*(vec3 const &v, component const s) {
    return {v.x * s, v.y * s, v.z * s};
}

__device__ __host__ vec3 operator*(component const s, vec3 const &v) {
    return {v.x * s, v.y * s, v.z * s};
}

__device__ __host__ vec3 operator/(vec3 const &v, component const s) {
    return v * (1 / s);
}

__device__ __host__ void operator/=(vec3 &v, component const s) {
    v.x /= s;
    v.y /= s;
    v.z /= s;
}

__device__ __host__ vec3 cross(vec3 const &a, vec3 const &b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
    };
}

__device__ __host__ component dot(vec3 const &a, vec3 const &b) {
    return a.x * b.x
           + a.y * b.y
           + a.z * b.z;
}

__device__ __host__ component length_squared(vec3 const &v) {
    return dot(v, v);
}

__device__ __host__ component length(vec3 const &v) {
    return sqrt(length_squared(v));
}

__device__ __host__ vec3 normalize(vec3 const &v) {
    return rsqrt(dot(v, v)) * v;
}


std::ostream &operator<<(std::ostream &out, vec3 const &v) {
    return out << std::format("({}, {}, {})", v.x, v.y, v.z);
}

#endif