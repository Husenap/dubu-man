cmake_minimum_required(VERSION 3.28)
project(dubu_man CUDA)

set(CMAKE_CUDA_STANDARD 26)


add_executable(dubu_man
        dubu-man/main.cu
        dubu-man/util/timer.cuh
        dubu-man/util/checks.cuh
        dubu-man/linalg/vec3.cuh
        dubu-man/linalg/ray.cuh
        dubu-man/linalg/camera.cuh
        dubu-man/linalg/camera.cu
        dubu-man/linalg/interval.cuh
        dubu-man/linalg/interval.cu
        dubu-man/linalg/trig.cuh
        dubu-man/hittable/hittable.cuh
        dubu-man/hittable/hit_record.cuh
        dubu-man/hittable/sphere.cuh
        dubu-man/hittable/hittable_list.cuh
        dubu-man/material/material.cuh
        dubu-man/material/lambertian.cuh
        dubu-man/material/metal.cuh
        dubu-man/material/dielectric.cuh
        dubu-man/material/dielectric.cu
        dubu-man/material/lambertian.cu
        dubu-man/material/metal.cu
)

set_target_properties(dubu_man PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if (USE_OIDN)
    if (DEFINED ENV{OIDN_CMAKE_DIR})
        include($ENV{OIDN_CMAKE_DIR}/OpenImageDenoiseConfig.cmake)
        target_link_libraries(dubu_man OpenImageDenoise)
        add_compile_definitions(USE_OIDN)
    else ()
        message(FATAL_ERROR "OIDN_CMAKE_DIR environment variable not set!")
    endif ()
endif ()
