/**
 * @file    cuda_many_rhs.cpp
 * @brief   GPU-accelerated distributed TDMA (many RHS, multi-vector) example using CuPaScaL_TDMA and MPI.
 * @details Demonstrates solving a distributed multi-RHS tridiagonal system on GPUs, comparing results with the
 *          CPU-based PaScaL_TDMA solver for validation.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "pascal_tdma.cuh"
#include "pascal_tdma.hpp"
#include "cuda_env.hpp"

/**
 * @brief Entry point for the CuPaScaL_TDMA many-RHS GPU example.
 *
 * Initializes MPI and CUDA, prepares multi-vector problem data, runs CPU and GPU solvers, and compares accuracy.
 */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Problem size
    const int nx = 8, ny = 40, nz = 160;
    const int N = nx * ny * nz;

    // Tridiagonal coefficients and RHS
    std::vector<double> a_h(nx, -1.0), b_h(nx, 4.0), c_h(nx, -1.0), d_h(N);

    for (int i = 0; i < N; i++) d_h[i] = std::sin(i);

    // Initialize CUDA environment
    CudaEnv::initialize();
    if (CudaEnv::isCudaAwareMPI()) {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is available.\n";
    } else {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is NOT available.\n";
    }

    // Allocate GPU memory
    double *a_d, *b_d, *c_d, *d_d;
    cudaMalloc(&a_d, nx * sizeof(double));
    cudaMalloc(&b_d, nx * sizeof(double));
    cudaMalloc(&c_d, nx * sizeof(double));
    cudaMalloc(&d_d, N * sizeof(double));

    // Copy host data to GPU
    cudaMemcpy(a_d, a_h.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // ===== CPU reference solve =====
    PaScaL_TDMA::PTDMAPlanManyRHS plan_cpu;
    plan_cpu.create(nx, ny * nz, MPI_COMM_WORLD, PaScaL_TDMA::TDMAType::Cyclic);
    PaScaL_TDMA::PTDMASolverManyRHS::solve(plan_cpu, a_h, b_h, c_h, d_h);
    plan_cpu.destroy();

    // ===== GPU solve =====
    CuPaScaL_TDMA::CuPTDMAPlanManyRHS plan_gpu;
    plan_gpu.create(nx, ny, nz, MPI_COMM_WORLD, CuPaScaL_TDMA::TDMAType::Cyclic);
    CuPaScaL_TDMA::CuPTDMASolverManyRHS::cuSolve(plan_gpu, a_d, b_d, c_d, d_d);
    plan_gpu.destroy();

    // Copy solution from device to host
    std::vector<double> d_h_out(N);
    cudaMemcpy(d_h_out.data(), d_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute error
    double error = 0.0;
    for (int i = 0; i < N; i++) error += std::abs(d_h[i] - d_h_out[i]);

    if (!rank)
        std::cout << "Avg. RMS error = " << std::sqrt(error / nx / ny / nz) << std::endl;

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(d_d);
    MPI_Finalize();

    return 0;
}