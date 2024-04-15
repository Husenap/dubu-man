#pragma once

#include "hittable.cuh"

namespace dubu_man {

    class hittable_list : public hittable {
        hittable **m_list{};
        size_t m_list_size{};
    public:
        __device__ hittable_list() {}

        __device__ hittable_list(hittable **list, size_t n) : m_list(list), m_list_size(n) {}

        __device__ ~hittable_list() override {
            for (size_t i = 0; i < m_list_size; ++i) {
                delete m_list[i];
            }
            delete[] m_list;
        }

        __device__ bool hit(ray const &r, interval ray_t, hit_record &rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;
            float closest_so_far = ray_t.max;

            for (size_t i = 0; i < m_list_size; ++i) {
                if (m_list[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
            return hit_anything;
        }
    };

} // dubu_man