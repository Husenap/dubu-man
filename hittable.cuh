#pragma once

#include <thrust/device_vector.h>

#include "vec3.cuh"
#include "ray.cuh"

namespace dubu_man {
    struct hit_record {
        point3 p;
        vec3 normal;
        float t;
        bool front_face;

        __device__ void set_face_normal(ray const &r, vec3 const &outward_normal) {
            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
    };

    class hittable {
    public:
        __device__ virtual ~hittable() {};

        __device__ virtual bool hit(ray const &r, float ray_tmin, float ray_tmax, hit_record &rec) const = 0;
    };

    class sphere : public hittable {
        point3 m_center;
        float m_radius;
    public:
        __device__ sphere() {}

        __device__ sphere(point3 center, float radius) : m_center(center), m_radius(radius) {}

        __device__ bool hit(ray const &r, float ray_tmin, float ray_tmax, hit_record &rec) const override {
            const auto oc = r.origin() - m_center;
            const auto a = length_squared(r.direction());
            const auto half_b = dot(oc, r.direction());
            const auto c = length_squared(oc) - m_radius * m_radius;
            const auto discriminant = half_b * half_b - a * c;

            if (discriminant < 0) return false;

            const auto sqrtd = sqrt(discriminant);

            auto root = (-half_b - sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root) {
                root = (-half_b + sqrtd) / a;
                if (root <= ray_tmin || ray_tmax <= root) {
                    return false;
                }
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            const auto outward_normal = (rec.p - m_center) / m_radius;
            rec.set_face_normal(r, outward_normal);

            return true;
        }
    };

    class hittable_list : public hittable {
        hittable **m_list;
        size_t m_list_size;
    public:
        __device__ hittable_list() {}

        __device__ ~hittable_list() {
            delete[] m_list;
        }

        __device__ hittable_list(hittable **list, size_t n) : m_list(list), m_list_size(n) {}

        __device__ bool hit(ray const &r, float ray_tmin, float ray_tmax, hit_record &rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;
            float closest_so_far = ray_tmax;
            for (size_t i = 0; i < m_list_size; ++i) {
                if (m_list[i]->hit(r, ray_tmin, closest_so_far, temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
            return hit_anything;
        }

    };

} // dubu_man