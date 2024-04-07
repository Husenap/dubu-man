#pragma once

#include "linalg/vec3.cuh"

namespace dubu_man {
    constexpr size_t IMAGE_WIDTH = 1024;
    constexpr float ASPECT_RATIO = 2.0f;
    constexpr size_t IMAGE_HEIGHT = static_cast<size_t>(IMAGE_WIDTH / ASPECT_RATIO);
    constexpr size_t NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
    constexpr size_t SAMPLES_PER_PIXEL = 20;
    constexpr size_t MAX_BOUNCES = 20;

    constexpr float VIEWPORT_HEIGHT = 2.0f;
    constexpr float VIEWPORT_WIDTH = std::max(1.0f, VIEWPORT_HEIGHT * (static_cast<float>(IMAGE_WIDTH) / IMAGE_HEIGHT));

    constexpr float FOCAL_LENGTH = 1.0f;
} // dubu_man
