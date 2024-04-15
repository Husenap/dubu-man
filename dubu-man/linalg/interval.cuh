#pragma once

namespace dubu_man {
    struct interval {
        float min, max;

        constexpr __host__ __device__ interval() : min(INFINITY), max(-INFINITY) {}

        constexpr __host__ __device__ interval(float min, float max) : min(min), max(max) {}

        [[nodiscard]] constexpr __host__ __device__ float size() const {
            return max - min;
        }

        [[nodiscard]] constexpr __host__ __device__ bool contains(float x) const {
            return min <= x && x <= max;
        }

        [[nodiscard]] constexpr __host__ __device__ bool surrounds(float x) const {
            return min < x && x < max;
        }

        [[nodiscard]] constexpr __host__ __device__ float clamp(float x) const {
            return x < min
                   ? min
                   : x > max
                     ? max
                     : x;
        }

        static const interval empty, universe;
    };
} // dubu_man