cmake_minimum_required(VERSION 3.28)
project(dubu_man CUDA)

set(CMAKE_CUDA_STANDARD 26)

add_executable(dubu_man main.cu
        float3_extension.cuh
        ray.cuh
        camera.cuh
        settings.cuh
)

set_target_properties(dubu_man PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
