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
#include "linalg/camera.cuh"
#include "util/timer.cuh"
#include "hittable/hittable.cuh"
#include "hittable/sphere.cuh"
#include "hittable/hittable_list.cuh"
#include "material/lambertian.cuh"
#include "material/dielectric.cuh"
#include "material/metal.cuh"

namespace dubu_man {
    struct PixelData {
        vec3 color;
        vec3 albedo;
        vec3 normal;
    };

    __global__ void render_init(const camera *cam, curandState *rand_state) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= cam->image_width || j >= cam->image_height) return;
        const auto pixel_index = j * cam->image_width + i;

        curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    }

    __host__ __device__ float linear_to_srgb(float value) {
        return value < 0.0031308f ? 12.92f * value : 1.055f * pow(value, 1.0f / 2.4f) - 0.055f;
    }

    __device__ color ray_color(ray const &r, const camera *cam, hittable *world, curandState &rand_state) {
        ray cur_ray = r;
        color cur_attenuation = color{1};

        for (size_t b = 0; b <= cam->max_bounces; ++b) {
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

    __global__ void render(PixelData framebuffer[], size_t framebuffer_pitch,
                           const camera *cam, hittable **world, curandState *rand_state) {
        for (unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
             py < cam->image_height; py += blockDim.y * gridDim.y) {
            auto *row = (PixelData *) ((char *) framebuffer + framebuffer_pitch * py);
            for (unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
                 px < cam->image_width; px += blockDim.x * gridDim.x) {
                const auto pixel_index = py * cam->image_width + px;

                auto local_rand_state = rand_state[pixel_index];

                color col{};
                color albedo{};
                color normal{};

                for (size_t s = 0; s < cam->samples_per_pixel; ++s) {
                    const auto r = cam->get_ray(px, py, local_rand_state);
                    col = col + ray_color(r, cam, *world, local_rand_state);

                    hit_record rec;
                    if ((*world)->hit(r, interval{0.001f, INFINITY}, rec)) {
                        normal = normal + rec.normal;
                        albedo = albedo + rec.material->get_albedo(rec);
                    }
                }

                col = col * cam->pixel_samples_scale;
                col.x = linear_to_srgb(col.x);
                col.y = linear_to_srgb(col.y);
                col.z = linear_to_srgb(col.z);

                row[px].color = col;
                row[px].albedo = albedo * cam->pixel_samples_scale;
                row[px].normal = normalize(normal);
            }
        }
    }

    __global__ void create_world_1(hittable **d_world) {
        constexpr size_t N = 5;
        auto **const list = new hittable *[N];
        list[0] = new sphere({0, -100.5f, -1}, 100, new lambertian({0.8f, 0.8f, 0.0f}));
        list[1] = new sphere({0, 0, -1.2f}, 0.5f, new lambertian({0.1f, 0.2f, 0.5f}));
        list[2] = new sphere({-1, 0, -1}, 0.5f, new dielectric(1.5f));
        list[3] = new sphere({-1, 0, -1}, 0.4f, new dielectric(1.0f / 1.5f));
        list[4] = new sphere({1, 0, -1}, 0.5f, new metal({0.8f, 0.6f, 0.2f}, 1.0f));
        *d_world = new hittable_list(list, N);
    }

    __global__ void create_world_2(hittable **d_world) {
        curandState rand_state;
        curand_init(1984, 0, 0, &rand_state);

        constexpr int M = 11;
        constexpr int MM = (M * 2 + 1);
        constexpr int N = MM * MM + 4;
        auto **const list = new hittable *[N];

        for (int a = -M; a <= M; ++a) {
            for (int b = -M; b <= M; ++b) {
                const auto i = (a + M) * MM + (b + M);
                const auto choose_mat = curand_uniform(&rand_state);
                const vec3 center{(float) a + 0.9f * curand_uniform(&rand_state), 0.2f,
                                  (float) b + 0.9f * curand_uniform(&rand_state)};

                if (choose_mat < 0.8f) {
                    const auto albedo = color::random(rand_state) * color::random(rand_state);
                    list[i] = new sphere(center, 0.2, new lambertian(albedo));
                } else if (choose_mat < 0.95f) {
                    const auto albedo = color::random(0.5f, 1.0f, rand_state);
                    const auto fuzz = curand_uniform(&rand_state) * 0.5f;
                    list[i] = new sphere(center, 0.2, new metal(albedo, fuzz));
                } else {
                    list[i] = new sphere(center, 0.2, new dielectric(1.5f));
                }
            }
        }

        list[MM * MM + 0] = new sphere({0, -1000, 0}, 1000.0f, new lambertian({0.5f, 0.5f, 0.5f}));
        list[MM * MM + 1] = new sphere({0, 1, 0}, 1.0f, new dielectric(1.5f));
        list[MM * MM + 2] = new sphere({-4, 1, 0}, 1.0f, new lambertian({0.4f, 0.2f, 0.1f}));
        list[MM * MM + 3] = new sphere({4, 1, 0}, 1.0f, new metal({0.7f, 0.6f, 0.5f}, 0.0f));

        *d_world = new hittable_list(list, N);
    }

    __global__ void free_world(hittable **d_world) {
        delete *d_world;
    }

    void run() {
        int device_id;
        cudaCheck(cudaGetDevice(&device_id));
        int sm_count;
        cudaCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));

        // Create camera
        hittable **d_world;
        cudaCheck(cudaMalloc(&d_world, sizeof(hittable *)));

        std::cout << "Scenes:\n"
                     "1. three balls front\n"
                     "2. final render in-one-weekend" << std::endl;
        int sceneId = 0;
        while (!(sceneId >= 1 && sceneId <= 2)) {
            std::cout << "Select scene: ";
            std::cin >> sceneId;
        }

        camera const *cam{};

        switch (sceneId) {
            case 1:
            default:
                cam = new camera{{.image_width = 1024,
                                         .aspect_ratio=2.0f,
                                         .samples_per_pixel=10,
                                         .max_bounces=10,
                                 }};
                create_world_1<<<1, 1>>>(d_world);
                break;
            case 2:
                cam = new camera{{.image_width = 600,
                                         .aspect_ratio = 2.0f,
                                         .samples_per_pixel = 10,
                                         .max_bounces = 10,

                                         .vfov = 20,
                                         .look_from = {13, 2, 3},
                                         .look_at = {0, 0, 0},
                                         .vup={0, 1, 0},

                                         .defocus_angle=0.6f,
                                         .focus_dist = 10.0f,

                                 }};
                create_world_2<<<1, 1>>>(d_world);
                break;
        }

        camera *d_cam;
        cudaCheck(cudaMalloc(&d_cam, sizeof(camera)));
        cudaCheck(cudaMemcpy(d_cam, cam, sizeof(camera), cudaMemcpyHostToDevice));

        const auto IMAGE_WIDTH = cam->image_width;
        const auto IMAGE_HEIGHT = cam->image_height;
        const auto NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

        // Allocate framebuffers to render to
        auto framebuffer = std::vector<PixelData>(NUM_PIXELS);
        auto color_framebuffer = std::vector<vec3>(NUM_PIXELS);
        auto albedo_framebuffer = std::vector<vec3>(NUM_PIXELS);
        auto normal_framebuffer = std::vector<vec3>(NUM_PIXELS);

        // Allocate random state
        curandState *d_rand_state;
        cudaCheck(cudaMalloc(&d_rand_state, NUM_PIXELS * sizeof(curandState)));

        { // Render
            size_t framebuffer_pitch;
            PixelData *d_framebuffer;
            cudaCheck(cudaMallocPitch(&d_framebuffer, &framebuffer_pitch, sizeof(PixelData) * IMAGE_WIDTH,
                                      IMAGE_HEIGHT));

            {
                const auto dim_block = dim3(32, 32, 1);
                const auto dim_grid = dim3(ceil(static_cast<float>(IMAGE_WIDTH) / static_cast<float>(dim_block.x)),
                                           ceil(static_cast<float>(IMAGE_HEIGHT) / static_cast<float>(dim_block.y)),
                                           1);
                timer t("Render Init");
                render_init<<<dim_grid, dim_block>>>(d_cam, d_rand_state);
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaDeviceSynchronize());
            }
            {
                const auto dim_block = dim3(16, 16, 1);
                timer t("Render");
                render<<<sm_count, dim_block>>>(d_framebuffer, framebuffer_pitch, d_cam, d_world, d_rand_state);
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaDeviceSynchronize());
            }

            cudaMemcpy2D(framebuffer.data(),
                         sizeof(PixelData) * IMAGE_WIDTH, d_framebuffer, framebuffer_pitch,
                         sizeof(PixelData) * IMAGE_WIDTH, IMAGE_HEIGHT, cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < framebuffer.size(); ++i) {
                color_framebuffer[i] = framebuffer[i].color;
                albedo_framebuffer[i] = framebuffer[i].albedo;
                normal_framebuffer[i] = framebuffer[i].normal;
            }


            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaFree(d_framebuffer));
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

        const auto images_to_save = std::vector<std::pair<std::string_view, vec3 *>>{
                {"image.png", color_framebuffer.data()},
                {"image_normal.png", normal_framebuffer.data()},
                {"image_albedo.png", albedo_framebuffer.data()},
#ifdef USE_OIDN
                {"image_oidn_denoised.png", (vec3 *) color_buffer.getData()},
                {"image_oidn_normal.png", (vec3 *) normal_buffer.getData()},
                {"image_oidn_albedo.png", (vec3 *) albedo_buffer.getData()},
#endif
        };

        for (const auto &[path, output]: images_to_save) { // Save color_framebuffer to file
            timer t(std::format("Writing image: {}", path));
            auto data = std::vector<uchar3>(NUM_PIXELS, {0, 0, 0});
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = {static_cast <unsigned char>( std::clamp(output[i].x, 0.0f, 0.999f) * 256),
                           static_cast <unsigned char>( std::clamp(output[i].y, 0.0f, 0.999f) * 256),
                           static_cast <unsigned char>( std::clamp(output[i].z, 0.0f, 0.999f) * 256)};
            }

            stbi_write_png(path.data(),
                           static_cast<int>(IMAGE_WIDTH), static_cast<int>(IMAGE_HEIGHT),
                           3, data.data(), static_cast<int>(IMAGE_WIDTH * sizeof(data[0])));
        }

        // Deallocate
        cudaCheck(cudaDeviceSynchronize());
        free_world<<<1, 1>>>(d_world);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaFree(d_cam));
        cudaCheck(cudaFree(d_world));
        cudaCheck(cudaFree(d_rand_state));
        delete cam;
    }
}

int main() {
    cudaFree(nullptr);
    dubu_man::run();
    return 0;
}