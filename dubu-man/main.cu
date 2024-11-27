#include <algorithm>
#include <iostream>
#include <vector>

#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef USE_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../thirdparty/stb_image_write.h"
#include "renderer/kernels.cuh"
#include "scenes/scenes.cuh"
#include "util/checks.cuh"
#include "util/timer.cuh"

#define SAVE_PNG_FILES 0

namespace dubu_man {

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {}

void run() {
  int device_id;
  cudaCheck(cudaGetDevice(&device_id));
  int sm_count;
  cudaCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));

  std::cout << "Scenes:\n"
               "1. three balls front\n"
               "2. final render in-one-weekend"
            << std::endl;
  int sceneId = 1;
  while (!(sceneId >= 1 && sceneId <= 2)) {
    std::cout << "Select scene: ";
    std::cin >> sceneId;
  }

  camera const* cam{};
  hittable2*    d_world;
  cudaCheck(cudaMalloc(&d_world, sizeof(hittable2)));

  switch (sceneId) {
  case 1:
  default:
    cam = new camera{{
        .image_width       = 600,
        .aspect_ratio      = 2.0f,
        .samples_per_pixel = 1,
        .max_bounces       = 20,
    }};
    create_world_1<<<1, 1>>>(d_world);
    break;
  case 2:
    cam = new camera{{
        .image_width       = 600,
        .aspect_ratio      = 2.0f,
        .samples_per_pixel = 1,
        .max_bounces       = 10,

        .vfov      = 20,
        .look_from = {13, 2, 3},
        .look_at   = {0, 0, 0},
        .vup       = {0, 1, 0},

        .defocus_angle = 0.6f,
        .focus_dist    = 10.0f,

    }};
    create_world_2<<<1, 1>>>(d_world);
    break;
  }

  camera* d_cam;
  cudaCheck(cudaMalloc(&d_cam, sizeof(camera)));
  cudaCheck(cudaMemcpy(d_cam, cam, sizeof(camera), cudaMemcpyHostToDevice));

  const auto IMAGE_WIDTH  = cam->image_width;
  const auto IMAGE_HEIGHT = cam->image_height;
  const auto NUM_PIXELS   = IMAGE_WIDTH * IMAGE_HEIGHT;

  // Allocate framebuffers to render to
  auto framebuffer        = std::vector<PixelData>(NUM_PIXELS);
  auto color_framebuffer  = std::vector<vec3>(NUM_PIXELS);
  auto albedo_framebuffer = std::vector<vec3>(NUM_PIXELS);
  auto normal_framebuffer = std::vector<vec3>(NUM_PIXELS);

  // Allocate random state
  curandState* d_rand_state;
  cudaCheck(cudaMalloc(&d_rand_state, NUM_PIXELS * sizeof(curandState)));

  { // Render
    size_t     framebuffer_pitch;
    PixelData* d_framebuffer;
    cudaCheck(cudaMallocPitch(&d_framebuffer, &framebuffer_pitch, sizeof(PixelData) * IMAGE_WIDTH, IMAGE_HEIGHT));

    {
      const auto dim_block = dim3(32, 32, 1);
      const auto dim_grid  = dim3(ceil(static_cast<float>(IMAGE_WIDTH) / static_cast<float>(dim_block.x)),
                                 ceil(static_cast<float>(IMAGE_HEIGHT) / static_cast<float>(dim_block.y)),
                                 1);
      timer      t("Render Init");
      render_init<<<dim_grid, dim_block>>>(d_cam, d_rand_state);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
    }
    {
      const auto dim_block = dim3(16, 16, 1);
      timer      t("Render");
      render<<<sm_count, dim_block>>>(d_framebuffer, framebuffer_pitch, d_cam, d_world, d_rand_state);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
    }

    cudaMemcpy2D(framebuffer.data(),
                 sizeof(PixelData) * IMAGE_WIDTH,
                 d_framebuffer,
                 framebuffer_pitch,
                 sizeof(PixelData) * IMAGE_WIDTH,
                 IMAGE_HEIGHT,
                 cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < framebuffer.size(); ++i) {
      color_framebuffer[i]  = framebuffer[i].color;
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
  oidn::BufferRef color_buffer  = oidn_device.newBuffer(NUM_PIXELS * sizeof(vec3));
  oidn::BufferRef albedo_buffer = oidn_device.newBuffer(NUM_PIXELS * sizeof(vec3));
  oidn::BufferRef normal_buffer = oidn_device.newBuffer(NUM_PIXELS * sizeof(vec3));

  // Copy framebuffers to OIDN buffers
  cudaCheck(cudaMemcpy(color_buffer.getData(), color_framebuffer.data(), NUM_PIXELS * sizeof(vec3), cudaMemcpyHostToHost));
  cudaCheck(cudaMemcpy(normal_buffer.getData(), normal_framebuffer.data(), NUM_PIXELS * sizeof(vec3), cudaMemcpyHostToHost));
  cudaCheck(cudaMemcpy(albedo_buffer.getData(), albedo_framebuffer.data(), NUM_PIXELS * sizeof(vec3), cudaMemcpyHostToHost));

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
  const char* errorMessage;
  if (oidn_device.getError(errorMessage) != oidn::Error::None) std::cerr << "Error: " << errorMessage << std::endl;
#endif

#if SAVE_PNG_FILES
  const auto images_to_save = std::vector<std::pair<std::string_view, vec3*>>{
      {"image.png", color_framebuffer.data()},
      {"image_normal.png", normal_framebuffer.data()},
      {"image_albedo.png", albedo_framebuffer.data()},
#ifdef USE_OIDN
      {"image_oidn_denoised.png", (vec3*)color_buffer.getData()},
      {"image_oidn_normal.png", (vec3*)normal_buffer.getData()},
      {"image_oidn_albedo.png", (vec3*)albedo_buffer.getData()},
#endif
  };

  for (const auto& [path, output] : images_to_save) { // Save color_framebuffer to file
    timer t(std::format("Writing image: {}", path));
    auto  data = std::vector<uchar3>(NUM_PIXELS, {0, 0, 0});
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = {static_cast<unsigned char>(std::clamp(output[i].x, 0.0f, 0.999f) * 256),
                 static_cast<unsigned char>(std::clamp(output[i].y, 0.0f, 0.999f) * 256),
                 static_cast<unsigned char>(std::clamp(output[i].z, 0.0f, 0.999f) * 256)};
    }

    stbi_write_png(path.data(),
                   static_cast<int>(IMAGE_WIDTH),
                   static_cast<int>(IMAGE_HEIGHT),
                   3,
                   data.data(),
                   static_cast<int>(IMAGE_WIDTH * sizeof(data[0])));
  }
#endif

  // Deallocate
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaFree(d_cam));
  cudaCheck(cudaFree(d_world));
  cudaCheck(cudaFree(d_rand_state));
  delete cam;

  GLFWwindow* window;
  {
    // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    const auto VIEWPORT_WIDTH  = 1920;
    const auto VIEWPORT_HEIGHT = static_cast<int>(VIEWPORT_WIDTH * IMAGE_HEIGHT / IMAGE_WIDTH);
    window                     = glfwCreateWindow(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, "CUDA Particles", nullptr, nullptr);
    glfwSetKeyCallback(window, key_callback);
    glfwMakeContextCurrent(window);

    // Initialize OpenGL
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
      std::cerr << "Failed to initialize GLAD" << std::endl;
      exit(-1);
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glfwSwapInterval(0);
  }

  cudaSetDevice(0);

  double time           = glfwGetTime();
  int    frame          = 0;
  double delta_time_acc = 0.0;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    const auto current_time = glfwGetTime();
    const auto frame_time   = current_time - time;
    time                    = current_time;

    delta_time_acc += frame_time;
    ++frame;
    if (delta_time_acc >= 1.0) {
      const auto delta_time_per_frame = delta_time_acc / frame;
      std::cout << "frame time: " << delta_time_per_frame * 1'000.0 << "ms" << std::endl;
      std::cout << "fps: " << (1.0 / delta_time_per_frame) << std::endl;
      delta_time_acc -= 1.0;
      frame = 0;
    }

    const auto delta_time = std::min(frame_time, 0.02);

    { // Update frame
      /*
      Particle* particles;
      cudaGraphicsMapResources(1, &particlesVBO_CUDA);
      size_t num_bytes;
      cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&particles), &num_bytes, particlesVBO_CUDA);
      const dim3 dim_block = 1024;
      const dim3 dim_grid  = (NUM_PARTICLES + dim_block.x - 1) / dim_block.x;
      update_particles<<<dim_grid, dim_block>>>( particles, d_board_state, d_block_positions, block_positions.size(),
      static_cast<float>(time), static_cast<float>(delta_time)); cudaGraphicsUnmapResources(1, &particlesVBO_CUDA);
      */
    }

    glClearColor(0.05f, 0.06f, 0.07f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // draw framebuffer

    glfwSwapBuffers(window);
  }

  { // Destroy GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
  }
}
} // namespace dubu_man

int main() {
  cudaFree(nullptr);
  dubu_man::run();
  return 0;
}