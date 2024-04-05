#pragma once

namespace dubu_man {
    class cuda_timer {
        cudaEvent_t m_start{}, m_stop{};
        std::string_view m_label;
    public:
        explicit cuda_timer(std::string_view label) : m_label(label) {
            cudaEventCreate(&m_start);
            cudaEventRecord(m_start, nullptr);
        }

        ~cuda_timer() {
            cudaEventCreate(&m_stop);
            cudaEventRecord(m_stop, nullptr);
            cudaEventSynchronize(m_stop);
            float elapsed_time;
            cudaEventElapsedTime(&elapsed_time, m_start, m_stop);
            std::clog << std::format("{}: {}ms", m_label, elapsed_time) << std::endl;
        }
    };
} // dubu_man