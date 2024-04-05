#pragma once

#include <iostream>
#include <format>

#include <curand_kernel.h>

struct vec3 {
    constexpr __host__ __device__ vec3() {}

    constexpr __host__ __device__ vec3(float s) : x(s), y(s), z(s) {}

    constexpr __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    float x{}, y{}, z{};

    __device__ static vec3 random(curandState &rand_state) {
        return {curand_uniform(&rand_state), curand_uniform(&rand_state), curand_uniform(&rand_state)};
    }

    __device__ static vec3 random(float min, float max, curandState &rand_state) {
        return {min + (max - min) * curand_uniform(&rand_state),
                min + (max - min) * curand_uniform(&rand_state),
                min + (max - min) * curand_uniform(&rand_state)};
    }

    __device__ static vec3 random_in_unit_sphere(curandState &rand_state);

    __device__ static vec3 random_unit_vector(curandState &rand_state);

    __device__ static vec3 random_on_hemisphere(vec3 const &normal, curandState &rand_state);
};

using point3 = vec3;
using color = vec3;

static __device__ __host__ vec3 operator+(vec3 const &a, vec3 const &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static __device__ __host__ vec3 operator-(vec3 const &a, vec3 const &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

static __device__ __host__ vec3 operator-(vec3 const &a) {
    return {-a.x, -a.y, -a.z};
}

static __device__ __host__ vec3 operator*(vec3 const &v, float const s) {
    return {v.x * s, v.y * s, v.z * s};
}

static __device__ __host__ vec3 operator*(float const s, vec3 const &v) {
    return {v.x * s, v.y * s, v.z * s};
}

static __device__ __host__ vec3 operator/(vec3 const &v, float const s) {
    return v * (1 / s);
}

static __device__ __host__ void operator/=(vec3 &v, float const s) {
    v.x /= s;
    v.y /= s;
    v.z /= s;
}

static __device__ __host__ vec3 cross(vec3 const &a, vec3 const &b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
    };
}

static __device__ __host__ float dot(vec3 const &a, vec3 const &b) {
    return a.x * b.x
           + a.y * b.y
           + a.z * b.z;
}

static __device__ __host__ float length_squared(vec3 const &v) {
    return dot(v, v);
}

static __device__ __host__ float length(vec3 const &v) {
    return sqrt(length_squared(v));
}

static __device__ __host__ vec3 normalize(vec3 const &v) {
    return rsqrt(dot(v, v)) * v;
}

static std::ostream &operator<<(std::ostream &out, vec3 const &v) {
    return out << std::format("({}, {}, {})", v.x, v.y, v.z);
}

inline __device__ vec3 vec3::random_in_unit_sphere(curandState &rand_state) {
    vec3 p{};
    do {
        p = vec3::random(-1.f, 1.f, rand_state);
    } while (length_squared(p) >= 1.0f);
    return p;
}

inline __device__ vec3 vec3::random_unit_vector(curandState &rand_state) {
    return normalize(vec3::random_in_unit_sphere(rand_state));
}

inline __device__ vec3 vec3::random_on_hemisphere(vec3 const &normal, curandState &rand_state) {
    const auto on_unit_sphere = vec3::random_unit_vector(rand_state);
    if (dot(on_unit_sphere, normal) >= 0.0f) {
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}
