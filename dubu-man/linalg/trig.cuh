#pragma once

namespace dubu_man {

    constexpr float pi = 3.1415926535897932385f;

    inline __device__ __host__ constexpr float degrees_to_radians(float degrees){
        return degrees * pi / 180.0f;
    }

} // dubu_man