#pragma once

#include "hittable.cuh"

namespace dubu_man {

    class sphere : public hittable {
        point3 m_center;
        float m_radius;
        material *m_material;
    public:
        __device__ sphere() {}

        __device__ ~sphere() override {
            printf("~sphere(), ");
            delete m_material;
        }

        __device__ sphere(point3 center, float radius, material *material) : m_center(center), m_radius(radius),
                                                                             m_material(material) {}

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
            rec.material = m_material;

            return true;
        }
    };

} // dubu_man