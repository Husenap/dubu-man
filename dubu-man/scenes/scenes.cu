#include "scenes.cuh"
#include "util/random.cuh"

namespace dubu_man {
__global__ void create_world_1(hittable* d_world) {
  constexpr size_t N    = 5;
  auto             list = new hittable[N]{};

  list[0] = hittable::make_sphere({0, -100.5f, -1}, 100, material::make_lambertian({0.8f, 0.8f, 0.0f}));
  list[1] = hittable::make_sphere({0, 0, -1.2f}, 0.5f, material::make_lambertian({0.1f, 0.2f, 0.5f}));
  list[2] = hittable::make_sphere({-1, 0, -1}, 0.5f, material::make_dielectric(1.5f));
  list[3] = hittable::make_sphere({-1, 0, -1}, 0.4f, material::make_dielectric(1.0f / 1.5f));
  list[4] = hittable::make_sphere({1, 0, -1}, 0.5f, material::make_metal({0.8f, 0.6f, 0.2f}, 1.0f));

  *d_world = hittable::make_hittable_list(list, N);
}

__global__ void create_world_2(hittable* d_world) {
  constexpr int M    = 11;
  constexpr int MM   = (M * 2 + 1);
  constexpr int N    = MM * MM + 4;
  auto* const   list = new hittable[N]{};

  curandState rand_state;
  curand_init(1984, 0, 0, &rand_state);

  for (int a = -M; a <= M; ++a) {
    for (int b = -M; b <= M; ++b) {
      const auto i          = (a + M) * MM + (b + M);
      const auto choose_mat = random01(rand_state);
      const vec3 center{(float)a + 0.9f * random01(rand_state), 0.2f, (float)b + 0.9f * random01(rand_state)};

      if (choose_mat < 0.8f) {
        const auto albedo = vec3{random01(rand_state), random01(rand_state), random01(rand_state)} *
                            vec3{random01(rand_state), random01(rand_state), random01(rand_state)};
        list[i] = hittable::make_sphere(center, 0.2, material::make_lambertian(albedo));
      } else if (choose_mat < 0.95f) {
        const auto albedo =
            color(random_range(0.5f, 1.0f, rand_state), random_range(0.5f, 1.0f, rand_state), random_range(0.5f, 1.0f, rand_state));
        const auto fuzz = random_range(0.0f, 0.5f, rand_state);
        list[i]         = hittable::make_sphere(center, 0.2, material::make_metal(albedo, fuzz));
      } else {
        list[i] = hittable::make_sphere(center, 0.2, material::make_dielectric(1.5f));
      }
    }
  }

  list[MM * MM + 0] = hittable::make_sphere({0, -1000.0f, 0}, 1000.0f, material::make_lambertian({0.5f, 0.5f, 0.5f}));
  list[MM * MM + 1] = hittable::make_sphere({0, 1, 0}, 1.0f, material::make_dielectric(1.5f));
  list[MM * MM + 2] = hittable::make_sphere({-4, 1, 0}, 1.0f, material::make_lambertian({0.4f, 0.2f, 0.1f}));
  list[MM * MM + 3] = hittable::make_sphere({4, 1, 0}, 1.0f, material::make_metal({0.7f, 0.6f, 0.5f}, 0.0f));

  *d_world = hittable::make_hittable_list(list, N);
}

} // namespace dubu_man