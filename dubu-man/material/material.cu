#include "../hittable/hit_record.cuh"
#include "material.cuh"

namespace dubu_man {

__host__ __device__ material material::make_lambertian(const color& albedo) {
  return {.type = material_type::lambertian, .lambertian = {.albedo = albedo}};
}

__host__ __device__ material material::make_metal(const color& albedo, float fuzz) {
  return {.type = material_type::metal, .metal = {.albedo = albedo, .fuzz = fuzz}};
}

__host__ __device__ material material::make_dielectric(float ior) {
  return {.type = material_type::dielectric, .dielectric = {.ior = ior}};
}

__device__ bool
material::scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState& rand_state) const {
  if (type == material_type::lambertian) {
    return scatter_lambertian(r_in, rec, attenuation, scattered, rand_state);
  } else if (type == material_type::metal) {
    return scatter_metal(r_in, rec, attenuation, scattered, rand_state);
  } else {
    return scatter_dielectric(r_in, rec, attenuation, scattered, rand_state);
  }
}

__device__ color const& material::get_albedo(const hit_record& rec) const {
  if (type == material_type::lambertian) {
    return lambertian.albedo;
  } else if (type == material_type::metal) {
    return metal.albedo;
  } else {
    static const color white{1.0f};
    return white;
  }
}

__device__ bool
material::scatter_lambertian(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState& rand_state) const {
  auto scatter_direction = rec.normal + vec3::random_unit_vector(rand_state);
  if (near_zero(scatter_direction)) {
    scatter_direction = rec.normal;
  }

  scattered   = ray(rec.p, scatter_direction);
  attenuation = lambertian.albedo;

  return true;
}

__device__ bool
material::scatter_metal(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState& rand_state) const {
  const auto reflected = reflect(normalize(r_in.direction()), rec.normal);
  scattered            = ray(rec.p, reflected + metal.fuzz * vec3::random_unit_vector(rand_state));
  attenuation          = metal.albedo;
  return (dot(scattered.direction(), rec.normal) > 0);
}

__device__ bool
material::scatter_dielectric(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState& rand_state) const {
  attenuation = color{1.0f};

  const auto ior            = rec.front_face ? (1.0f / dielectric.ior) : dielectric.ior;
  const auto unit_direction = normalize(r_in.direction());
  const auto cos_theta      = fmin(dot(-unit_direction, rec.normal), 1.0f);
  const auto sin_theta      = sqrt(1.0f - cos_theta * cos_theta);

  bool cannot_refract = ior * sin_theta > 1.0f;
  vec3 direction;

  if (cannot_refract || reflectance(cos_theta, ior) > curand_uniform(&rand_state)) {
    direction = reflect(unit_direction, rec.normal);
  } else {
    direction = refract(unit_direction, rec.normal, ior);
  }

  scattered = ray(rec.p, direction);

  return true;
}
} // namespace dubu_man