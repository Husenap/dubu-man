#include "interval.cuh"

namespace dubu_man {
    const interval interval::empty = interval{INFINITY, -INFINITY};
    const interval interval::universe = interval{-INFINITY, INFINITY};
}