/**
 * @file    cuda_many.cpp
 * @brief   GPU-accelerated distributed TDMA (many RHS) example using CuPaScaL_TDMA and MPI.
 * @details Demonstrates how to set up, solve, and compare CPU and GPU solvers for a large TDMA system
 *          in parallel with CUDA-aware MPI.
 *
 * Usage:
 *      mpirun -n <num_procs> ./cuda_many_example
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "pascal_tdma.cuh"
#include "pascal_tdma.hpp"
#include "cuda_env.hpp"

/**
 * @brief Program entry point for the CuPaScaL_TDMA many-RHS GPU example.
 *
 * Initializes MPI and CUDA, prepares problem data, runs CPU and GPU solvers, and compares results.
 */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Problem size
    const int nx = 4, ny = 40, nz = 160;
    const int N = nx * ny * nz;

    // Host-side system coefficients and RHS
    std::vector<double> a_h(N, -1.0), b_h(N, 4.0), c_h(N, -1.0), d_h(N);

    // Initialize CUDA environment and check for CUDA-aware MPI
    CudaEnv::initialize();
    if (CudaEnv::isCudaAwareMPI()) {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is available.\n";
    } else {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is NOT available.\n";
    }

    // Initialize RHS with a known function
    for (int i = 0; i < N; i++) d_h[i] = std::sin(i);

    // Allocate GPU memory
    double *a_d, *b_d, *c_d, *d_d;
    cudaMalloc((void**)&a_d, N * sizeof(double));
    cudaMalloc((void**)&b_d, N * sizeof(double));
    cudaMalloc((void**)&c_d, N * sizeof(double));
    cudaMalloc((void**)&d_d, N * sizeof(double));

    // Copy host data to GPU
    cudaMemcpy(a_d, a_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // ===== CPU reference solve =====
    PaScaL_TDMA::PTDMAPlanMany plan_cpu;
    plan_cpu.create(nx, ny * nz, MPI_COMM_WORLD, PaScaL_TDMA::TDMAType::Standard);
    PaScaL_TDMA::PTDMASolverMany::solve(plan_cpu, a_h, b_h, c_h, d_h);
    plan_cpu.destroy();

    // ===== GPU solve =====
    CuPaScaL_TDMA::CuPTDMAPlanMany plan_gpu;
    plan_gpu.create(nx, ny, nz, MPI_COMM_WORLD, CuPaScaL_TDMA::TDMAType::Standard);
    CuPaScaL_TDMA::CuPTDMASolverMany::cuSolve(plan_gpu, a_d, b_d, c_d, d_d);
    plan_gpu.destroy();

    // Copy solution back to host
    std::vector<double> d_h_out(N);
    cudaMemcpy(d_h_out.data(), d_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute total error
    double error = 0.0;
    for (int i = 0; i < N; i++) error += std::abs(d_h[i] - d_h_out[i]);

    if (!rank)
        std::cout << "Avg. RMS error = " << std::sqrt(error / nx / ny / nz) << std::endl;

    // Free GPU memory and finalize
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(d_d);

    MPI_Finalize();
    return 0;
}