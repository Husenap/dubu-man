cmake_minimum_required(VERSION 3.28)
project(dubu_man
        LANGUAGES CXX C CUDA)

set(CMAKE_CUDA_STANDARD 26)

include("cmake/glfw.cmake")
add_subdirectory("cmake/glad")

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
        dubu-man/material/material.cuh
        dubu-man/material/material.cu
        dubu-man/renderer/kernels.cu
        dubu-man/scenes/scenes.cu
        dubu-man/util/random.cuh
)
target_include_directories(dubu_man PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/dubu-man/")
target_link_libraries(dubu_man glfw glad)
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
