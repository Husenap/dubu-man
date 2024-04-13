#include <iostream>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef USE_OIDN

#include <OpenImageDenoise/oidn.hpp>

#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../thirdparty/stb_image_write.h"

#include "util/checks.cuh"
#include "linalg/vec3.cuh"
#include "settings.cuh"
#include "linalg/camera.cuh"
#include "util/timer.cuh"
#include "hittable/hittable.cuh"
#include "hittable/sphere.cuh"
#include "hittable/hittable_list.cuh"
#include "material/material.cuh"

namespace dubu_man {
    __global__ void render_init(curandState *rand_state) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= IMAGE_WIDTH || j >= IMAGE_HEIGHT) return;
        const auto pixel_index = j * IMAGE_WIDTH + i;

        curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    }

    __host__ __device__ float linear_to_srgb(float value) {
        return value < 0.0031308f ? 12.92f * value : 1.055f * pow(value, 1.0f / 2.4f) - 0.055f;
    }

    __host__ __device__ float linear_to_gamma(float linear_component) {
        return pow(linear_component, 1.0f / 2.2f);
    }

    __device__ color ray_color(ray const &r, hittable *world, curandState &rand_state) {
        ray cur_ray = r;
        color cur_attenuation = color{1};

        for (size_t b = 0; b <= MAX_BOUNCES; ++b) {
            hit_record rec;
            if (world->hit(cur_ray, interval{0.001f, INFINITY}, rec)) {
                color attenuation;
                if (!rec.material->scatter(cur_ray, rec, attenuation, cur_ray, rand_state))
                    return color{0};
                cur_attenuation = cur_attenuation * attenuation;
            } else {
                const auto unit_direction = normalize(cur_ray.direction());
                const auto a = 0.5f * (unit_direction.y + 1.0f);
                color c = (1.0f - a) * color{1.0f, 1.0f, 1.0f} + a * color{0.5f, 0.7f, 1.0f};
                return cur_attenuation * c;
            }
        }
        return color{0};
    }

    __global__ void
    render(vec3 color_framebuffer[], vec3 albedo_framebuffer[], vec3 normal_framebuffer[], size_t framebuffer_pitch,
           camera const cam,
           hittable **world, curandState *rand_state) {
        for (unsigned int py = blockIdx.y * blockDim.y + threadIdx.y; py < IMAGE_HEIGHT; py += blockDim.y * gridDim.y) {
            vec3 *color_row = (vec3 *) ((char *) color_framebuffer + framebuffer_pitch * py);
            vec3 *albedo_row = (vec3 *) ((char *) albedo_framebuffer + framebuffer_pitch * py);
            vec3 *normal_row = (vec3 *) ((char *) normal_framebuffer + framebuffer_pitch * py);
            for (unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
                 px < IMAGE_WIDTH; px += blockDim.x * gridDim.x) {
                const auto pixel_index = py * IMAGE_WIDTH + px;
                auto local_rand_state = rand_state[pixel_index];

                color col{};
                color albedo{};
                color normal{};

                for (size_t s = 0; s < SAMPLES_PER_PIXEL; ++s) {
                    const auto r = cam.get_sub_pixel_ray(px, py, local_rand_state);
                    col = col + ray_color(r, *world, local_rand_state);

                    hit_record rec;
                    if ((*world)->hit(r, interval{0.001f, INFINITY}, rec)) {
                        normal = normal + rec.normal;
                        albedo = albedo + rec.material->get_albedo(rec);
                    }
                }

                rand_state[pixel_index] = local_rand_state;

                col = col / float(SAMPLES_PER_PIXEL);
                col.x = linear_to_srgb(col.x);
                col.y = linear_to_srgb(col.y);
                col.z = linear_to_srgb(col.z);

                color_row[px] = col;
                albedo_row[px] = albedo / float(SAMPLES_PER_PIXEL);
                normal_row[px] = normalize(normal);
            }
        }
    }

    __global__ void create_world(hittable **d_world) {
        auto **const list = new hittable *[4];
        list[0] = new sphere({0, -100.5f, -1}, 100, new lambertian({0.8f, 0.8f, 0.0f}));
        list[1] = new sphere({0, 0, -1}, 0.5f, new lambertian({0.7f, 0.3f, 0.3f}));
        list[2] = new sphere({-1, 0, -1}, 0.5f, new metal({0.8f, 0.8f, 0.8f}, 0.3f));
        list[3] = new sphere({1, 0, -1}, 0.5f, new metal({0.8f, 0.6f, 0.2f}, 1.0f));
        *d_world = new hittable_list(list, 4);
    }

    __global__ void free_world(hittable **d_world) {
        printf("deleting world \n");
        delete *d_world;
    }

    void run() {
        int device_id;
        cudaCheck(cudaGetDevice(&device_id));
        int sm_count;
        cudaCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));

        // Allocate framebuffers to render to
        auto color_framebuffer = std::vector<vec3>(NUM_PIXELS);
        auto albedo_framebuffer = std::vector<vec3>(NUM_PIXELS);
        auto normal_framebuffer = std::vector<vec3>(NUM_PIXELS);

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
            size_t framebuffer_pitch;
            vec3 *d_color_framebuffer;
            cudaCheck(cudaMallocPitch(&d_color_framebuffer, &framebuffer_pitch, sizeof(vec3) * IMAGE_WIDTH,
                                      IMAGE_HEIGHT));
            vec3 *d_albedo_framebuffer;
            cudaCheck(cudaMallocPitch(&d_albedo_framebuffer, &framebuffer_pitch, sizeof(vec3) * IMAGE_WIDTH,
                                      IMAGE_HEIGHT));
            vec3 *d_normal_framebuffer;
            cudaCheck(cudaMallocPitch(&d_normal_framebuffer, &framebuffer_pitch, sizeof(vec3) * IMAGE_WIDTH,
                                      IMAGE_HEIGHT));

            {
                const auto dim_block = dim3(32, 32, 1);
                const auto dim_grid = dim3(ceil(IMAGE_WIDTH / static_cast<float>(dim_block.x)),
                                           ceil(IMAGE_HEIGHT / static_cast<float>(dim_block.y)),
                                           1);
                timer t("Render Init");
                render_init<<<dim_grid, dim_block>>>(d_rand_state);
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaDeviceSynchronize());
            }
            {
                const auto dim_block = dim3(16, 16, 1);
                timer t("Render");
                render<<<sm_count, dim_block>>>(d_color_framebuffer,
                                                                   d_albedo_framebuffer,
                                                                   d_normal_framebuffer,
                                                                   framebuffer_pitch,
                                                                   cam,
                                                                   d_world, d_rand_state);
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaDeviceSynchronize());
            }

            cudaMemcpy2D(color_framebuffer.data(), sizeof(vec3) * IMAGE_WIDTH, d_color_framebuffer, framebuffer_pitch,
                         sizeof(vec3) * IMAGE_WIDTH, IMAGE_HEIGHT, cudaMemcpyDeviceToHost);
            cudaMemcpy2D(albedo_framebuffer.data(), sizeof(vec3) * IMAGE_WIDTH, d_albedo_framebuffer, framebuffer_pitch,
                         sizeof(vec3) * IMAGE_WIDTH, IMAGE_HEIGHT, cudaMemcpyDeviceToHost);
            cudaMemcpy2D(normal_framebuffer.data(), sizeof(vec3) * IMAGE_WIDTH, d_normal_framebuffer, framebuffer_pitch,
                         sizeof(vec3) * IMAGE_WIDTH, IMAGE_HEIGHT, cudaMemcpyDeviceToHost);
            cudaCheck(cudaDeviceSynchronize());

            cudaCheck(cudaFree(d_color_framebuffer));
            cudaCheck(cudaFree(d_albedo_framebuffer));
            cudaCheck(cudaFree(d_normal_framebuffer));
        }

#ifdef USE_OIDN
        // OIDN (Open Image Denoise) Device
        oidn::DeviceRef oidn_device = oidn::newCUDADevice(-1, nullptr);
        oidn_device.commit();

        // OIDN Buffers
        oidn::BufferRef color_buffer = oidn_device.newBuffer(NUM_PIXELS * sizeof(vec3));
        oidn::BufferRef albedo_buffer = oidn_device.newBuffer(NUM_PIXELS * sizeof(vec3));
        oidn::BufferRef normal_buffer = oidn_device.newBuffer(NUM_PIXELS * sizeof(vec3));

        // Copy framebuffers to OIDN buffers
        cudaCheck(cudaMemcpy(color_buffer.getData(), color_framebuffer.data(),
                             NUM_PIXELS * sizeof(vec3), cudaMemcpyHostToHost));
        cudaCheck(cudaMemcpy(normal_buffer.getData(), normal_framebuffer.data(),
                             NUM_PIXELS * sizeof(vec3), cudaMemcpyHostToHost));
        cudaCheck(cudaMemcpy(albedo_buffer.getData(), albedo_framebuffer.data(),
                             NUM_PIXELS * sizeof(vec3), cudaMemcpyHostToHost));

        // OIDN Beauty filter
        oidn::FilterRef filter = oidn_device.newFilter("RT");
        filter.setImage("color", color_buffer, oidn::Format::Float3, IMAGE_WIDTH, IMAGE_HEIGHT);
        filter.setImage("albedo", albedo_buffer, oidn::Format::Float3, IMAGE_WIDTH, IMAGE_HEIGHT);
        filter.setImage("normal", normal_buffer, oidn::Format::Float3, IMAGE_WIDTH, IMAGE_HEIGHT);
        filter.setImage("output", color_buffer, oidn::Format::Float3, IMAGE_WIDTH, IMAGE_HEIGHT);
        filter.set("hdr", false);
        filter.set("srgb", true);
        filter.set("cleanAux", true);
        filter.set("quality", OIDN_QUALITY_HIGH);
        filter.commit();

        // OIDN Albedo filter
        oidn::FilterRef albedo_filter = oidn_device.newFilter("RT");
        albedo_filter.setImage("albedo", albedo_buffer, oidn::Format::Float3, IMAGE_WIDTH, IMAGE_HEIGHT);
        albedo_filter.setImage("output", albedo_buffer, oidn::Format::Float3, IMAGE_WIDTH, IMAGE_HEIGHT);
        albedo_filter.commit();

        // OIDN Normal filter
        oidn::FilterRef normal_filter = oidn_device.newFilter("RT");
        normal_filter.setImage("normal", normal_buffer, oidn::Format::Float3, IMAGE_WIDTH, IMAGE_HEIGHT);
        normal_filter.setImage("output", normal_buffer, oidn::Format::Float3, IMAGE_WIDTH, IMAGE_HEIGHT);
        normal_filter.commit();
        {
            timer t("OIDN Albedo Pre-filter");
            albedo_filter.execute();
        }
        {
            timer t("OIDN Normal Pre-filter");
            normal_filter.execute();
        }
        {
            timer t("OIDN Denoising");
            filter.execute();
        }
        oidn_device.sync();
        const char *errorMessage;
        if (oidn_device.getError(errorMessage) != oidn::Error::None)
            std::cerr << "Error: " << errorMessage << std::endl;
#endif

        const auto images_to_save = std::vector<std::tuple<std::string_view, vec3 *>>{
                {"image.png", color_framebuffer.data(),},
                {"image_normal.png", normal_framebuffer.data(),},
                {"image_albedo.png", albedo_framebuffer.data(),},
#ifdef USE_OIDN
                {"image_oidn_denoised.png", (vec3 *) color_buffer.getData(),},
                {"image_oidn_normal.png", (vec3 *) normal_buffer.getData(),},
                {"image_oidn_albedo.png", (vec3 *) albedo_buffer.getData(),},
#endif
        };

        for (const auto &[path, output]: images_to_save) { // Save color_framebuffer to file
            timer t(std::format("Writing image: {}", path));
            auto data = std::vector<uchar3>(NUM_PIXELS, {0, 0, 0});
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = {static_cast<unsigned char>( std::clamp(output[i].x, 0.0f, 0.999f) * 256),
                           static_cast<unsigned char>( std::clamp(output[i].y, 0.0f, 0.999f) * 256),
                           static_cast<unsigned char>( std::clamp(output[i].z, 0.0f, 0.999f) * 256)};
            }

            stbi_write_png(path.data(),
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
    }
}

int main() {
    // Initialize CUDA
    cudaFree(nullptr);

    dubu_man::run();

    cudaCheck(cudaDeviceReset());

    return 0;
}