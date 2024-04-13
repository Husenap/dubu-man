#pragma once

#include <iostream>
#include <format>

#include <curand_kernel.h>

struct vec3 {
    union {
        struct {
            float x, y, z;
        };
        float e[3]{0,0,0};
    };

    constexpr __host__ __device__ vec3() {}

    constexpr __host__ __device__ explicit vec3(float s) : x(s), y(s), z(s) {}

    constexpr __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    constexpr __host__ __device__ float operator[](int i) const { return e[i]; }

    constexpr __host__ __device__ float &operator[](int i) { return e[i]; }

    constexpr __host__ __device__ vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ vec3 &operator+=(const vec3 &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __host__ __device__ vec3 &operator*=(const vec3& v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    __host__ __device__ vec3 &operator*=(const float s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    __host__ __device__ vec3 &operator/=(const float s) {
        return *this *= 1 / s;
    }

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

static __device__ __host__ vec3 operator*(vec3 const &v, float const s) {
    return {v.x * s, v.y * s, v.z * s};
}

static __device__ __host__ vec3 operator*(vec3 const &a, vec3 const &b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

static __device__ __host__ vec3 operator*(float const s, vec3 const &v) {
    return {v.x * s, v.y * s, v.z * s};
}

static __device__ __host__ vec3 operator/(vec3 const &v, float const s) {
    return v * (1 / s);
}

static __device__ __host__ vec3 cross(vec3 const &a, vec3 const &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

static __device__ __host__ float dot(vec3 const &a, vec3 const &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
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

static __device__ __host__ bool near_zero(vec3 const &v) {
    constexpr auto eps = 1e-8f;
    return (fabs(v.x) < eps) && (fabs(v.y) < eps) && (fabs(v.z) < eps);
}

static __device__ __host__ vec3 reflect(vec3 const &i, vec3 const &n) {
    return i - 2.0f * n * dot(i, n);
}

static __device__ __host__ vec3 refract(vec3 const &uv, vec3 const &n, float etai_over_etat) {
    const auto cos_theta = fmin(-dot(uv, n), 1.0f);
    const vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    const vec3 r_out_parallel = -sqrt(fabs(1.0f - length_squared(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}
