/**
 * @file util.hpp
 * @brief Utility functions for parallel range computation and vector norms.
 *
 * This file provides functions to compute parallel distribution of ranges
 * among MPI ranks and to calculate vector norms.
*/

#pragma once

#include <algorithm> /**< For std::min */
#include <cmath>     /**< For std::sqrt */
#include <vector>    /**< For std::vector */

namespace Util { 

/**
 * @brief Computes the number of elements assigned to a given rank.
 *
 * Divides an inclusive range [first, last] across 'size' ranks.
 *
 * @param first The first index (inclusive).
 * @param last The last index (inclusive).
 * @param size Total number of ranks.
 * @param rank The rank for which to compute the block size.
 * @return Number of elements assigned to 'rank'.
 */
inline int para_range_n(int first, int last, int size, int rank) noexcept {
    const int n = last - first + 1;
    const int base = n / size;
    const int remainder = n % size;
    return base + (rank < remainder ? 1 : 0);
}

/**
 * @brief Computes the start and end indices for a given rank, and returns 
 * the number of elements assigned to 'rank'.
 *
 * Divides 'n' elements among 'size' ranks, setting output parameters
 * 'sta' and 'end'.
 *
 * @param n Total number of elements.
 * @param size Total number of ranks.
 * @param rank Rank for which to compute the range.
 * @param[out] sta Computed starting index for this rank.
 * @param[out] end Computed ending index for this rank.
 * @return Number of elements assigned to 'rank'.
 */
inline int para_range(int n, int size, int rank, int &sta, int &end) noexcept {
    const int base = n / size;
    const int remainder = n % size;
    sta = base * rank + std::min(rank, remainder);
    end = sta + base - 1 + (rank < remainder ? 1 : 0);
    return end - sta + 1;
}

/**
 * @brief Calculates the L2 (Euclidean) norm of a vector.
 *
 * @tparam T Numeric type of vector elements.
 * @param v Input vector.
 * @return L2 norm of the vector.
 */
template <class T> 
inline T norm2(const std::vector<T>& v) noexcept {
    T sum = 0.0;
    for (auto val : v) sum += val * val;
    return std::sqrt(sum);
}
} // namespace util