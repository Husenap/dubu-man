#include "hittable/hittable.cuh"
#include "kernels.cuh"
#include "linalg/camera.cuh"
#include "linalg/vec3.cuh"
#include "util/checks.cuh"
#include "util/random.cuh"

namespace dubu_man {
__global__ void render_init(const camera* cam, curandState* rand_state) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= cam->image_width || j >= cam->image_height) return;
  const auto pixel_index = j * cam->image_width + i;

  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__host__ __device__ float linear_to_srgb(float value) {
  return value < 0.0031308f ? 12.92f * value : 1.055f * pow(value, 1.0f / 2.4f) - 0.055f;
}

__device__ color ray_color(ray const& r, const camera* cam, hittable2 world, curandState& rand_state) {
  ray   cur_ray         = r;
  color cur_attenuation = color{1};

  for (size_t b = 0; b <= cam->max_bounces; ++b) {
    hit_record rec;
    if (world.hit(cur_ray, interval{0.001f, INFINITY}, rec)) {
      color attenuation;
      if (!rec.material.scatter(cur_ray, rec, attenuation, cur_ray, rand_state)) return color{0};
      cur_attenuation = cur_attenuation * attenuation;
    } else {
      const auto unit_direction = normalize(cur_ray.direction());
      const auto a              = 0.5f * (unit_direction.y + 1.0f);
      color      c              = (1.0f - a) * color{1.0f, 1.0f, 1.0f} + a * color{0.5f, 0.7f, 1.0f};
      return cur_attenuation * c;
    }
  }
  return color{0};
}

__global__ void
render(PixelData framebuffer[], size_t framebuffer_pitch, const camera* cam, hittable2* world, curandState* rand_state, int frame) {
  for (unsigned int py = blockIdx.y * blockDim.y + threadIdx.y; py < cam->image_height; py += blockDim.y * gridDim.y) {
    for (unsigned int px = blockIdx.x * blockDim.x + threadIdx.x; px < cam->image_width; px += blockDim.x * gridDim.x) {
      const auto pixel_index = py * cam->image_width + px;

      auto& local_rand_state = rand_state[pixel_index];

      color col{};
      color albedo{};
      color normal{};

      const auto r = cam->get_ray(px, py, local_rand_state);
      col          = col + ray_color(r, cam, *world, local_rand_state);

      hit_record rec;
      if (world->hit(r, interval{0.001f, INFINITY}, rec)) {
        normal = normal + rec.normal;
        albedo = albedo + rec.material.get_albedo(rec);
      }

      col.x = linear_to_srgb(col.x);
      col.y = linear_to_srgb(col.y);
      col.z = linear_to_srgb(col.z);

      auto& pixel = ((PixelData*)((char*)framebuffer + framebuffer_pitch * py))[px];

      const auto previous_decay = static_cast<float>(frame - 1) / static_cast<float>(frame);
      const auto current_decay  = 1.0f / static_cast<float>(frame);

      pixel.color = pixel.color * previous_decay + col * current_decay;
      pixel.albedo = pixel.albedo * previous_decay + albedo * current_decay;
      pixel.normal = pixel.normal * previous_decay + normalize(normal) * current_decay;
    }
  }
}

} // namespace dubu_man