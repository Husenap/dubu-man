#include <iostream>
#include <vector>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "thirdparty/stb_image_write.h"

#include "util.cuh"
#include "vec3.cuh"
#include "settings.cuh"
#include "camera.cuh"
#include "cuda_timer.cuh"
#include "hittable.cuh"

#include <curand_kernel.h>

namespace dubu_man {
    __device__ float linear_to_gamma(float linear_component) {
        return pow(linear_component, 1.0f/2.2f);
    }

    __device__ color ray_color(ray const &r, hittable *world, curandState &rand_state) {
        ray cur_ray = r;
        float cur_attenuation = 1.0f;

        for (size_t b = 0; b < MAX_BOUNCES; ++b) {
            hit_record rec;
            if (world->hit(cur_ray, 0.001f, INFINITY, rec)) {
                vec3 direction = rec.normal + vec3::random_unit_vector(rand_state);
                cur_attenuation *= 0.5f;
                cur_ray = ray(rec.p, direction);
            } else {
                const auto unit_direction = normalize(cur_ray.direction());
                const auto a = 0.5f * (unit_direction.y + 1.0f);
                color c = (1.0f - a) * color{1.0f, 1.0f, 1.0f} + a * color{0.5f, 0.7f, 1.0f};
                return cur_attenuation * c;
            }
        }
        return vec3{0};
    }

    __global__ void render_init(curandState *rand_state) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= IMAGE_WIDTH || j >= IMAGE_HEIGHT) return;
        const auto pixel_index = j * IMAGE_WIDTH + i;

        curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    }

    __global__ void render(vec3 framebuffer[], camera const cam, hittable **world, curandState *rand_state) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= IMAGE_WIDTH || j >= IMAGE_HEIGHT) return;
        const auto pixel_index = j * IMAGE_WIDTH + i;

        auto local_rand_state = rand_state[pixel_index];
        color col{0};
        for (size_t s = 0; s < SAMPLES_PER_PIXEL; ++s) {
            const auto r = cam.get_ray(i, j, local_rand_state);
            col = col + ray_color(r, *world, local_rand_state);
        }
        rand_state[pixel_index] = local_rand_state;
        col = col / float(SAMPLES_PER_PIXEL);
        col.x = linear_to_gamma(col.x);
        col.y = linear_to_gamma(col.y);
        col.z = linear_to_gamma(col.z);
        framebuffer[pixel_index] = col;
    }

    __global__ void create_world(hittable **d_world) {
        auto **const list = new hittable *[2];
        list[0] = new sphere({0, 0, -1}, 0.5f);
        list[1] = new sphere({0, -100.5f, -1}, 100);
        *d_world = new hittable_list(list, 2);
    }

    __global__ void free_world(hittable **d_world) {
        delete *d_world;
    }

    void run() {
        // Allocate framebuffer to render to
        vec3 *framebuffer;
        cudaCheck(cudaMallocManaged(&framebuffer, sizeof(vec3) * NUM_PIXELS));

        // Allocate random state
        curandState *d_rand_state;
        cudaCheck(cudaMalloc(&d_rand_state, NUM_PIXELS * sizeof(curandState)));

        // Create camera
        const camera cam{vec3{0, 0, 0}};

        // Allocate world
        hittable **d_world;
        cudaCheck(cudaMalloc(&d_world, sizeof(hittable *)));
        create_world<<<1, 1>>>(d_world);

        { // Render
            const auto block_dim = dim3(8, 8, 1);
            const auto grid_dim = dim3(ceil(IMAGE_WIDTH / static_cast<float>(block_dim.x)),
                                       ceil(IMAGE_HEIGHT / static_cast<float>(block_dim.y)),
                                       1);
            {
                cuda_timer timer("Render Init");
                render_init<<<grid_dim, block_dim>>>(d_rand_state);
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaDeviceSynchronize());
            }
            {
                cuda_timer timer("Render");
                render<<<grid_dim, block_dim>>>(framebuffer, cam, d_world, d_rand_state);
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaDeviceSynchronize());
            }
        }


        { // Save framebuffer to file
            auto data = std::vector<uchar3>(NUM_PIXELS, {0, 0, 0});
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = {static_cast<unsigned char>(std::clamp(framebuffer[i].x, 0.0f, 0.999f) * 256),
                           static_cast<unsigned char>(std::clamp(framebuffer[i].y, 0.0f, 0.999f) * 256),
                           static_cast<unsigned char>(std::clamp(framebuffer[i].z, 0.0f, 0.999f) * 256)};
            }

            stbi_write_png("image.png",
                           IMAGE_WIDTH,
                           IMAGE_HEIGHT,
                           3,
                           data.data(),
                           IMAGE_WIDTH * sizeof(data[0]));

        }

        // Deallocate
        cudaCheck(cudaDeviceSynchronize());
        free_world<<<1, 1>>>(d_world);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaFree(d_world));
        cudaCheck(cudaFree(d_rand_state));
        cudaCheck(cudaFree(framebuffer));
    }
}

int main() {
    dubu_man::run();
    return 0;
}