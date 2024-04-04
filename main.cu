#include <iostream>
#include <format>
#include <vector>
#include <algorithm>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "thirdparty/stb_image_write.h"

#include "float3_extension.cuh"
#include "settings.cuh"
#include "camera.cuh"

namespace dubu_man {
    __device__ component hit_sphere(point3 const &center, component radius, ray const &r) {
        const auto oc = r.origin() - center;
        const auto a = length_squared(r.direction());
        const auto half_b = dot(oc, r.direction());
        const auto c = length_squared(oc) - radius * radius;
        const auto discriminant = half_b * half_b - a * c;

        if (discriminant < 0) return -1.0;

        return (-half_b - sqrt(discriminant)) / a;
    }

    __device__ color ray_color(ray const &r) {
        constexpr auto sphere_center = point3{0, 0, -1};
        const auto t = hit_sphere(sphere_center, 0.5, r);
        if (t > 0) {
            const auto normal = normalize(r.at(t) - sphere_center);
            return normal * 0.5 + vec3{0.5, 0.5, 0.5};
        };

        const auto unit_direction = normalize(r.direction());
        const auto a = component(0.5) * (unit_direction.y + 1.0);
        return component(1.0 - a) * color{1.0, 1.0, 1.0} + component(a) * color{0.5, 0.7, 1.0};
    }

    __global__ void draw_pixel(vec3 output[IMAGE_WIDTH * IMAGE_HEIGHT], camera const cam) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i >= IMAGE_WIDTH || j >= IMAGE_HEIGHT) return;

        const auto r = cam.get_ray(i, j);
        output[j * IMAGE_WIDTH + i] = ray_color(r);
    }

    void run() {
        auto pixels = std::vector<vec3>(IMAGE_WIDTH * IMAGE_HEIGHT, {0, 0, 0});

        vec3 *d_pixels;
        cudaMalloc(&d_pixels, sizeof(pixels[0]) * pixels.size());

        const camera cam{vec3{0, 0, 0}};

        const auto threads_per_block = dim3(32, 32, 1);
        const auto num_blocks = dim3(
                ceil(IMAGE_WIDTH / static_cast<float>(threads_per_block.x)),
                ceil(IMAGE_HEIGHT / static_cast<float>(threads_per_block.y)),
                1
        );
        std::clog << "Rendering..." << std::endl;
        const auto t0 = std::chrono::high_resolution_clock::now();
        draw_pixel<<<num_blocks, threads_per_block>>>(d_pixels, cam);
        const auto t1 = std::chrono::high_resolution_clock::now();
        std::clog << std::format("Done: {}ms",
                                 static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                         t1 - t0).count()) / 1000.0)
                  << std::endl;

        cudaMemcpy(pixels.data(), d_pixels, sizeof(pixels[0]) * pixels.size(), cudaMemcpyDeviceToHost);

        {
            auto data = std::vector<uchar3>(pixels.size(), {0, 0, 0});
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = {
                        static_cast<unsigned char>(std::clamp(pixels[i].x, component(0.0), component(1.0)) * 255.999),
                        static_cast<unsigned char>(std::clamp(pixels[i].y, component(0.0), component(1.0)) * 255.999),
                        static_cast<unsigned char>(std::clamp(pixels[i].z, component(0.0), component(1.0)) * 255.999)};
            }

            stbi_write_png("normals.png",
                           IMAGE_WIDTH,
                           IMAGE_HEIGHT,
                           3,
                           data.data(),
                           IMAGE_WIDTH * sizeof(data[0]));
        }
    }
}

int main() {
    dubu_man::run();
    return 0;
}