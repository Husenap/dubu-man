#pragma once

#include <format>

#define cudaCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << std::format("CUDA Error[{}:{}]: {}", file, line, cudaGetErrorString(code)) << std::endl;
        cudaDeviceReset();
        exit(code);
    }
}