#pragma once

#include "hittable/hittable.cuh"

namespace dubu_man {

__global__ void create_world_1(hittable* d_world);
__global__ void create_world_2(hittable* d_world);

} // namespace dubu_man