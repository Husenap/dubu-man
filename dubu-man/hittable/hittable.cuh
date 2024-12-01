#pragma once

#include "../linalg/interval.cuh"
#include "../linalg/ray.cuh"
#include "../material/material.cuh"
#include "hit_record.cuh"

namespace dubu_man {
struct hittable {
  enum class hittable_type { sphere, hittable_list };

  hittable_type type;
  union {
    struct sphere_data {
      point3   center = point3(0);
      float    radius = 0.0f;
      material material;
    } sphere;
    struct hittable_list_data {
      hittable* list{};
      size_t    list_size{};
    } hittable_list;
  };

  static __host__ __device__ hittable make_sphere(point3 const& center, float radius, material const& material) {
    return {.type = hittable_type::sphere, .sphere = {.center = center, .radius = radius, .material = material}};
  }

  static __host__ __device__ hittable make_hittable_list(hittable* list, size_t list_size) {
    return {.type = hittable_type::hittable_list, .hittable_list = {.list = list, .list_size = list_size}};
  }

  __device__ bool hit(ray const& r, interval ray_t, hit_record& rec) const {
    if (type == hittable_type::hittable_list) {
      return hit_hittable_list(r, ray_t, rec);
    } else {
      return hit_sphere(r, ray_t, rec);
    }
  }

private:
  __device__ bool hit_sphere(ray const& r, interval ray_t, hit_record& rec) const {
    const auto oc = sphere.center - r.origin();
    const auto a  = length_squared(r.direction());
    const auto h  = dot(oc, r.direction());
    const auto c  = length_squared(oc) - sphere.radius * sphere.radius;

    const auto discriminant = h * h - a * c;
    if (discriminant < 0) return false;

    const auto sqrtd = sqrt(discriminant);

    auto root = (h - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
      root = (h + sqrtd) / a;
      if (!ray_t.surrounds(root)) {
        return false;
      }
    }

    rec.t                     = root;
    rec.p                     = r.at(rec.t);
    const auto outward_normal = (rec.p - sphere.center) / sphere.radius;
    rec.set_face_normal(r, outward_normal);
    rec.material = sphere.material;
    return true;
  }

  __device__ bool hit_hittable_list(ray const& r, interval ray_t, hit_record& rec) const {
    hit_record temp_rec;
    bool       hit_anything   = false;
    float      closest_so_far = ray_t.max;

    for (size_t i = 0; i < hittable_list.list_size; ++i) {
      if (hittable_list.list[i].hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
        hit_anything   = true;
        closest_so_far = temp_rec.t;
        rec            = temp_rec;
      }
    }
    return hit_anything;
  }
};

} // namespace dubu_man

static std::ostream& operator<<(std::ostream& out, dubu_man::hittable const& v) {
  switch (v.type) {
  case dubu_man::hittable::hittable_type::sphere:
    out << std::format("sphere: {}, {}", v.sphere.center, v.sphere.radius);
    break;
  case dubu_man::hittable::hittable_type::hittable_list:
    out << std::format("hittable_list: {} elements", v.hittable_list.list_size);
    break;
  }
  return out;
}
