#include <algorithm>
#include <iostream>
#include <vector>

#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef USE_OIDN
#include <OpenImageDenoise/oidn.hpp>
#include <cuda_gl_interop.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../thirdparty/stb_image_write.h"
#include "renderer/kernels.cuh"
#include "scenes/scenes.cuh"
#include "util/checks.cuh"
#include "util/timer.cuh"

namespace dubu_man {

const char* vertex_shader_source = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* fragment_shader_source = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D screenTexture;

void main() {
    FragColor = vec4(texture(screenTexture, TexCoord).rgb, 1.0);
}
)";

constexpr float PI = 3.1415926535898f;

// clang-format off
constexpr float quadVertices[] = {
  // positions     // texCoords
  -1.0f,  -1.0f,    0.0f, 1.0f,
  -1.0f, 1.0f,    0.0f, 0.0f,
  1.0f, 1.0f,    1.0f, 0.0f,

  -1.0f,  -1.0f,    0.0f, 1.0f,
  1.0f, 1.0f,    1.0f, 0.0f,
  1.0f,  -1.0f,    1.0f, 1.0f
};
// clang-format on

__global__ void copyToSurface(cudaSurfaceObject_t surface, float3* src, size_t width, size_t height) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const auto idx = y * width + x;
    float3     rgb = src[idx];
    surf2Dwrite(make_float4(rgb.x, rgb.y, rgb.z, 1.0f), surface, x * sizeof(float4), y);
  }
}

__global__ void extractBuffers(const PixelData* framebuffer,
                               size_t           framebuffer_pitch,
                               vec3*            color_buffer,
                               vec3*            albedo_buffer,
                               vec3*            normal_buffer,
                               size_t           width,
                               size_t           height) {
  const auto px = blockIdx.x * blockDim.x + threadIdx.x;
  const auto py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height) return;
  const auto idx = py * width + px;

  // Get the pixel data from the framebuffer
  const auto& pixel = ((const PixelData*)((const char*)framebuffer + framebuffer_pitch * py))[px];

  // Copy the data into the respective OIDN buffers
  color_buffer[idx]  = pixel.color;
  albedo_buffer[idx] = pixel.albedo;
  normal_buffer[idx] = pixel.normal;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

class camera_controller {
  vec3  pos               = {};
  float pitch             = {};
  float yaw               = -PI * 0.5f;
  float movement_speed    = 1.0f;
  float mouse_sensitivity = 0.002f;

  vec3 get_look_direction() {
    const float cp = std::cosf(pitch);
    const float cy = std::cosf(yaw);
    const float sp = std::sinf(pitch);
    const float sy = std::sinf(yaw);
    return normalize(vec3(cp * cy, sp, cp * sy));
  }

public:
  struct settings {
    float movement_speed    = 1.0f;
    float mouse_sensitivity = 0.002f;
  };
  camera_controller() = default;
  explicit camera_controller(const settings s, const camera::camera_settings cs)
      : pos(cs.look_from), movement_speed(s.movement_speed), mouse_sensitivity(s.mouse_sensitivity) {
    const auto d = normalize(cs.look_at - cs.look_from);
    pitch        = std::asinf(d.y);
    yaw          = std::atan2f(-d.x, d.z) + 0.5f * PI;
  }

  bool update(GLFWwindow* window, float dt) {
    vec3 target_velocity = {};
    { // Calculate Camera Movement
      float speed_multiplier = 1.0f;
      if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) speed_multiplier = 10.0f;
      if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL)) speed_multiplier = 0.1f;

      const float speed = movement_speed * speed_multiplier;

      const auto forward = get_look_direction();
      const auto right   = normalize(cross(forward, vec3(0, 1, 0)));
      const auto up      = normalize(cross(right, forward));

      if (glfwGetKey(window, GLFW_KEY_W)) target_velocity += forward;
      if (glfwGetKey(window, GLFW_KEY_S)) target_velocity -= forward;
      if (glfwGetKey(window, GLFW_KEY_D)) target_velocity += right;
      if (glfwGetKey(window, GLFW_KEY_A)) target_velocity -= right;
      if (glfwGetKey(window, GLFW_KEY_E)) target_velocity += up;
      if (glfwGetKey(window, GLFW_KEY_Q)) target_velocity -= up;

      pos += target_velocity * speed * dt;
    }

    float delta_yaw   = {};
    float delta_pitch = {};
    { // Calculate Camera Rotation
      double cursor_x, cursor_y;
      glfwGetCursorPos(window, &cursor_x, &cursor_y);
      static double previous_cursor_x     = cursor_x;
      static double previous_cursor_y     = cursor_y;
      static int    previous_mouse_active = GLFW_RELEASE;
      int           current_mouse_active  = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
      if (current_mouse_active && !previous_mouse_active) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
      }
      if (!current_mouse_active && previous_mouse_active) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
      }
      if (current_mouse_active) {
        delta_yaw   = static_cast<float>(cursor_x - previous_cursor_x);
        delta_pitch = static_cast<float>(previous_cursor_y - cursor_y);

        pitch = std::clamp(pitch + delta_pitch * mouse_sensitivity, -0.499f * PI, 0.499f * PI);
        yaw += delta_yaw * mouse_sensitivity;
        while (yaw < 0.0f) yaw += 2.0f * PI;
        while (yaw > 2.0f * PI) yaw -= 2.0f * PI;
      }
      previous_mouse_active = current_mouse_active;
      previous_cursor_x     = cursor_x;
      previous_cursor_y     = cursor_y;
    }

    return length_squared(target_velocity) > FLT_EPSILON || delta_yaw * delta_yaw + delta_pitch * delta_pitch > FLT_EPSILON;
  }

  camera get_camera(camera::camera_settings settings) {
    settings.look_from = pos;
    settings.look_at   = pos + get_look_direction();

    // std::cerr << "update camera: from: " << settings.look_from << ", to: " << settings.look_from << std::endl;

    return camera{settings};
  }
};

class app {
  // CUDA State
  int sm_count = {};

  // Scene State
  int               scene_id   = 2;
  hittable2*        d_world    = {};
  camera            cam        = {};
  camera*           d_cam      = {};
  camera_controller controller = {};

  // Framebuffer State
  size_t       framebuffer_pitch = {};
  PixelData*   d_framebuffer     = {};
  curandState* d_rand_state      = {};
  size_t       image_width       = {};
  size_t       image_height      = {};
  size_t       num_pixels        = {};

  // OIDN State
  oidn::DeviceRef oidn_device    = {};
  oidn::BufferRef color_buffer   = {};
  oidn::BufferRef albedo_buffer  = {};
  oidn::BufferRef normal_buffer  = {};
  oidn::FilterRef filter         = {};
  oidn::FilterRef albedo_filter  = {};
  oidn::FilterRef normal_filter  = {};
  bool            enable_denoise = true;

  // GLFW State
  GLFWwindow* window = {};

  // OpenGL State
  GLuint                texture      = {};
  cudaGraphicsResource* cudaResource = {};
  GLuint                quadVAO      = {};
  GLuint                quadVBO      = {};
  GLuint                program      = {};

public:
  app() {
    int device_id;
    cudaCheck(cudaGetDevice(&device_id));
    cudaCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));

    scene_prompt();
    init_render_state();
    init_oidn();
    init_glfw();
    init_opengl();
  }
  ~app() {
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaFree(d_rand_state));
    cudaCheck(cudaFree(d_framebuffer));
    cudaCheck(cudaFree(d_cam));
    cudaCheck(cudaFree(d_world));
  }

  // Select a scene to render
  void scene_prompt() {
    std::cout << "Scenes:\n"
                 "1. three balls front\n"
                 "2. final render in-one-weekend"
              << std::endl;
    while (!(scene_id >= 1 && scene_id <= 2)) {
      std::cout << "Select scene: ";
      std::cin >> scene_id;
    }

    cudaCheck(cudaMalloc(&d_world, sizeof(hittable2)));

    switch (scene_id) {
    case 1:
    default:
      cam = camera{{
          .image_width  = 600,
          .aspect_ratio = 2.0f,
          .max_bounces  = 10,
      }};
      create_world_1<<<1, 1>>>(d_world);
      break;
    case 2:
      cam = camera{{
          .image_width  = 600,
          .aspect_ratio = 2.0f,
          .max_bounces  = 10,

          .vfov      = 20,
          .look_from = {13, 2, 3},
          .look_at   = {0, 0, 0},
          .vup       = {0, 1, 0},

          .defocus_angle = 0.0f,
          .focus_dist    = 1.0f,
      }};
      create_world_2<<<1, 1>>>(d_world);
      break;
    }

    controller = camera_controller({}, cam.settings);

    cudaCheck(cudaMalloc(&d_cam, sizeof(camera)));
    cudaCheck(cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice));
  }

  // Initialize the render buffers
  void init_render_state() {
    image_width  = cam.image_width;
    image_height = cam.image_height;
    num_pixels   = image_width * image_height;

    // Allocate framebuffer to render to
    cudaCheck(cudaMallocPitch(&d_framebuffer, &framebuffer_pitch, sizeof(PixelData) * image_width, image_height));

    // Allocate a random state for each pixel in the framebuffer
    cudaCheck(cudaMalloc(&d_rand_state, num_pixels * sizeof(curandState)));

    { // Initialize random state for each pixel
      const auto dim_block = dim3(32, 32, 1);
      const auto dim_grid  = dim3(ceil(static_cast<float>(image_width) / static_cast<float>(dim_block.x)),
                                 ceil(static_cast<float>(image_height) / static_cast<float>(dim_block.y)),
                                 1);
      timer      t("Render Init");
      render_init<<<dim_grid, dim_block>>>(d_cam, d_rand_state);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
    }
  }

  // Initialize the Open Image Denoise device, buffers and filters
  void init_oidn() {
    // OIDN (Open Image Denoise) Device
    oidn_device = oidn::newCUDADevice(-1, nullptr);
    oidn_device.commit();

    // OIDN Buffers
    color_buffer  = oidn_device.newBuffer(num_pixels * sizeof(vec3));
    albedo_buffer = oidn_device.newBuffer(num_pixels * sizeof(vec3));
    normal_buffer = oidn_device.newBuffer(num_pixels * sizeof(vec3));

    // OIDN Beauty filter
    filter = oidn_device.newFilter("RT");
    filter.setImage("color", color_buffer, oidn::Format::Float3, image_width, image_height);
    filter.setImage("albedo", albedo_buffer, oidn::Format::Float3, image_width, image_height);
    filter.setImage("normal", normal_buffer, oidn::Format::Float3, image_width, image_height);
    filter.setImage("output", color_buffer, oidn::Format::Float3, image_width, image_height);
    filter.set("hdr", false);
    filter.set("srgb", true);
    filter.set("cleanAux", true);
    filter.set("quality", OIDN_QUALITY_HIGH);
    filter.commit();

    // OIDN Albedo filter
    albedo_filter = oidn_device.newFilter("RT");
    albedo_filter.setImage("albedo", albedo_buffer, oidn::Format::Float3, image_width, image_height);
    albedo_filter.setImage("output", albedo_buffer, oidn::Format::Float3, image_width, image_height);
    albedo_filter.commit();

    // OIDN Normal filter
    normal_filter = oidn_device.newFilter("RT");
    normal_filter.setImage("normal", normal_buffer, oidn::Format::Float3, image_width, image_height);
    normal_filter.setImage("output", normal_buffer, oidn::Format::Float3, image_width, image_height);
    normal_filter.commit();
  }

  // Initialize GLFW Window and GLAD
  void init_glfw() {
    {
      // Initialize GLFW
      glfwInit();
      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
      glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

      // Create window
      const auto VIEWPORT_WIDTH  = 1280;
      const auto VIEWPORT_HEIGHT = static_cast<int>(VIEWPORT_WIDTH * image_height / image_width);
      window                     = glfwCreateWindow(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, "dubu-man", nullptr, nullptr);
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
  }

  // Initialize OpenGL Texture, Buffers Shaders, and CUDAResource
  void init_opengl() {
    { // Create an OpenGL Texture
      glGenTextures(1, &texture);
      glBindTexture(GL_TEXTURE_2D, texture);
      glTexImage2D(
          GL_TEXTURE_2D, 0, GL_RGBA32F, static_cast<int>(image_width), static_cast<int>(image_height), 0, GL_RGBA, GL_FLOAT, nullptr);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glBindTexture(GL_TEXTURE_2D, 0);
    }

    std::vector<float> testData(image_width * image_height * 4, 0.5f); // White
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0, static_cast<int>(image_width), static_cast<int>(image_height), GL_RGBA, GL_FLOAT, testData.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    // Register the Texture with CUDA
    cudaCheck(cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

    { // Create Triangles for Fullscreen Quad
      glGenVertexArrays(1, &quadVAO);
      glGenBuffers(1, &quadVBO);

      glBindVertexArray(quadVAO);
      glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
      glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)nullptr);
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
      glBindVertexArray(0);
    }

    { // Create particle shader
      GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
      glShaderSource(vertexShader, 1, &vertex_shader_source, nullptr);
      glCompileShader(vertexShader);
      int  success;
      char infoLog[512];
      glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
      if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
      }

      GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
      glShaderSource(fragmentShader, 1, &fragment_shader_source, nullptr);
      glCompileShader(fragmentShader);
      glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
      if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
      }

      program = glCreateProgram();
      glAttachShader(program, vertexShader);
      glAttachShader(program, fragmentShader);
      glLinkProgram(program);
      glDeleteShader(vertexShader);
      glDeleteShader(fragmentShader);
    }
  }

  // Render a frame of the scene
  void render_frame(int frame) {
    {   // Render
      { // Render scene to framebuffer, 1 spp
        const auto dim_block = dim3(16, 16, 1);
        timer      t("Render");
        render<<<sm_count, dim_block>>>(d_framebuffer, framebuffer_pitch, d_cam, d_world, d_rand_state, frame);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
      }
    }

    // Copy framebuffer to OIDN buffers
    dim3 threads(32, 32);
    dim3 blocks((image_width + threads.x - 1) / threads.x, (image_height + threads.y - 1) / threads.y);
    extractBuffers<<<blocks, threads>>>((PixelData*)d_framebuffer,
                                        framebuffer_pitch,
                                        reinterpret_cast<vec3*>(color_buffer.getData()),
                                        reinterpret_cast<vec3*>(albedo_buffer.getData()),
                                        reinterpret_cast<vec3*>(normal_buffer.getData()),
                                        image_width,
                                        image_height);
    cudaCheck(cudaGetLastError());

    if (enable_denoise) {
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
      if (oidn_device.getError(errorMessage) != oidn::Error::None) {
        std::cerr << "Error: " << errorMessage << std::endl;
      }
    }
  }

  void run() {
    double time            = glfwGetTime();
    int    frame           = 0;
    int    recursive_frame = 0;
    double delta_time_acc  = 0.0;

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

      const auto delta_time = std::min(frame_time, 0.05);

      { // Update Camera
        if (controller.update(window, static_cast<float>(delta_time))) {
          cam = controller.get_camera(cam.settings);
          cudaCheck(cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice));
          recursive_frame = 0;
        }
      }

      {
        static int previous_toggle = false;
        int        current_toggle  = glfwGetKey(window, GLFW_KEY_TAB);
        if (current_toggle && !previous_toggle) {
          enable_denoise = !enable_denoise;
        }
        previous_toggle = current_toggle;
      }

      {
        // Render new ray traced frame
        render_frame(++recursive_frame);

        // Map the Texture in CUDA
        cudaCheck(cudaGraphicsMapResources(1, &cudaResource, nullptr));
        cudaArray* cudaArray;
        cudaCheck(cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0));

        cudaResourceDesc resDesc = {};
        resDesc.resType          = cudaResourceTypeArray;
        resDesc.res.array.array  = cudaArray;

        cudaSurfaceObject_t surface;
        cudaCreateSurfaceObject(&surface, &resDesc);

        cudaCheck(cudaGraphicsUnmapResources(1, &cudaResource, nullptr));
        cudaCheck(cudaGraphicsMapResources(1, &cudaResource, nullptr));

        // Copy Data from CUDA Buffer to OpenGL Texture
        dim3 threads(32, 32);
        dim3 blocks((image_width + threads.x - 1) / threads.x, (image_height + threads.y - 1) / threads.y);
        copyToSurface<<<blocks, threads>>>(surface, reinterpret_cast<float3*>(color_buffer.getData()), image_width, image_height);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        // Unmap the Texture
        cudaDestroySurfaceObject(surface);
        cudaCheck(cudaGraphicsUnmapResources(1, &cudaResource, nullptr));
        cudaCheck(cudaDeviceSynchronize());
      }

      { // Render
        glClearColor(0.05f, 0.06f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // draw framebuffer
        glUseProgram(program);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(program, "screenTexture"), 0);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
      }

      glfwSwapBuffers(window);
    }

    // Cleanup resources
    cudaGraphicsUnregisterResource(cudaResource);
    glDeleteTextures(1, &texture);

    { // Destroy GLFW
      glfwDestroyWindow(window);
      glfwTerminate();
    }
  }
};

} // namespace dubu_man

int main() {
  cudaFree(nullptr); // Initialize CUDA

  dubu_man::app app;
  app.run();

  return 0;
}