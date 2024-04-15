#include "interval.cuh"

namespace dubu_man {
    constexpr interval interval::empty = interval{INFINITY, -INFINITY};
    constexpr interval interval::universe = interval{-INFINITY, INFINITY};
}