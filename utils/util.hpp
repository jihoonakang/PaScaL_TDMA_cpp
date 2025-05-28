#pragma once
#include <cmath>

namespace Util { 

// Helper to compute number of rows per process
inline int para_range_n(int first, int last, int size, int rank) noexcept {
    const int n = last - first + 1;
    const int base = n / size;
    const int remainder = n % size;
    return base + (rank < remainder ? 1 : 0);

}

// Helper to compute the indices and the number of rows per process
inline int para_range(int n, int size, int rank, int &sta, int &end) noexcept {
    const int base = n / size;
    const int remainder = n % size;
    sta = base * rank + std::min(rank, remainder);
    end = sta + base - 1 + (rank < remainder ? 1 : 0);
    return end - sta + 1;
}

// Compute L2 norm
template <class T> 
inline T norm2(const std::vector<T>& v) {
    T sum = 0.0;
    for (auto val : v) sum += val * val;
    return std::sqrt(sum);
}
}