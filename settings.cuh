#ifndef DUBU_MAN_SETTINGS_CUH
#define DUBU_MAN_SETTINGS_CUH

#include "float3_extension.cuh"


namespace dubu_man {
    constexpr size_t IMAGE_WIDTH = 1024;
    constexpr component ASPECT_RATIO = 16.0 / 9.0;
    constexpr size_t IMAGE_HEIGHT = static_cast<size_t>(IMAGE_WIDTH / ASPECT_RATIO);

    constexpr component VIEWPORT_HEIGHT = 2.0;
    constexpr component VIEWPORT_WIDTH = VIEWPORT_HEIGHT * (static_cast<component>(IMAGE_WIDTH) / IMAGE_HEIGHT);

    constexpr component FOCAL_LENGTH = 1.0;
} // dubu_man

#endif