#pragma once

#include <format>
#include <cuda.h>

#define cudaCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << std::format("CUDA Error[{}:{}]: {}", file, line, cudaGetErrorString(code)) << std::endl;
        cudaDeviceReset();
        exit(code);
    }
}

#define cuCheck(ans) { cuAssert((ans), __FILE__, __LINE__); }

inline void cuAssert(CUresult code, const char *file, int line) {
    if (code != CUDA_SUCCESS) {
        const char *str;
        cuGetErrorString(code, &str);
        std::cerr << std::format("CU Error[{}:{}]: {}", file, line, str) << std::endl;
        cudaDeviceReset();
        exit(code);
    }
}
