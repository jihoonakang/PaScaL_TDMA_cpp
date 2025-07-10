/**
 * @file TDMASolver.cuh
 * @brief Header file for GPU-accelerated tridiagonal system solvers using CUDA.
 *
 */

#pragma once

#include <cuda_runtime.h>
#include <cassert>

/**
 * @class cuTDMASolver
 * @brief CUDA-based solvers for multiple tridiagonal systems.
 *
 */
class cuTDMASolver {
public:
    /// Solve many independent tridiagonal systems on the GPU.
    static void cuMany(
        const double* a_d,    ///< lower-diagonal coeffs (nx*ny*nz)
        const double* b_d,    ///< main diagonal coeffs
        double*       c_d,    ///< upper-diagonal coeffs (in/out)
        double*       d_d,    ///< RHS vectors (in/out)
        int           nx,     ///< rows per system
        int           ny,     ///< systems in Y-dimension
        int           nz,     ///< systems in Z-dimension
        cudaStream_t  stream = 0 ///< CUDA stream
    ) noexcept;

    /// Solve many cyclic tridiagonal systems on the GPU.
    static void cuManyCyclic(
        const double* a_d,    ///< lower-diagonal coeffs
        const double* b_d,    ///< main diagonal coeffs
        double*       c_d,    ///< upper-diagonal coeffs (in/out)
        double*       d_d,    ///< RHS vectors (in/out)
        int           nx,     ///< rows per system
        int           ny,     ///< systems in Y-dimension
        int           nz,     ///< systems in Z-dimension
        cudaStream_t  stream = 0 ///< CUDA stream
    ) noexcept;

    /// Solve batched systems with shared diagonals (multiple RHS) on the GPU.
    static void cuManyRHS(
        const double* a_d,    ///< lower-diagonal (shared)
        const double* b_d,    ///< main diagonal (shared)
        double*       c_d,    ///< upper-diagonal coeffs (in/out)
        double*       d_d,    ///< RHS buffer (in/out)
        int           nx,     ///< rows per system
        int           ny,     ///< number of RHS per row
        int           nz,     ///< systems in Z-dimension
        cudaStream_t  stream = 0 ///< CUDA stream
    ) noexcept;

    /// Solve batched cyclic systems with shared diagonals on the GPU.
    static void cuManyRHSCyclic(
        const double* a_d,    ///< lower-diagonal coeffs
        const double* b_d,    ///< main diagonal coeffs
        double*       c_d,    ///< upper-diagonal coeffs (in/out)
        double*       d_d,    ///< RHS vectors (in/out)
        int           nx,     ///< rows per system
        int           ny,     ///< number of RHS per row
        int           nz,     ///< systems in Z-dimension
        cudaStream_t  stream = 0 ///< CUDA stream
    ) noexcept;
};