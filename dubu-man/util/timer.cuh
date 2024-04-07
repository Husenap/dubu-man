#pragma once

#include <chrono>
#include <utility>

namespace dubu_man {
    class timer {
        std::chrono::time_point<std::chrono::steady_clock> m_start{};
        std::string m_label;
    public:
        explicit timer(std::string label) : m_label(std::move(label)), m_start(std::chrono::high_resolution_clock::now()) {}

        ~timer() {
            const auto stop = std::chrono::high_resolution_clock::now();
            float elapsed_time =
                    static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(stop - m_start).count()) /
                    1000.0f;
            std::clog << std::format("{}: {}ms", m_label, elapsed_time) << std::endl;
        }
    };
} // dubu_man