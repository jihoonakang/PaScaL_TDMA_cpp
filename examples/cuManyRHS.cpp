/**
 * @file cuManyRHS.cpp
 * @brief Example for GPU-accelerated distributed TDMA (many right-hand sides, multi-vector) using cuPaScaL_TDMA and MPI.
 *
 * This example demonstrates the use of cuPaScaL_TDMA for solving a distributed multi-RHS tridiagonal system on GPUs,
 * comparing the results to the CPU (PaScaL_TDMA) solver for validation.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "PaScaL_TDMA.cuh"
#include "PaScaL_TDMA.hpp"
#include "cudaEnv.hpp"

/**
 * @brief Main entry point for the cuPaScaL_TDMA many RHS GPU example.
 *
 * Initializes MPI and CUDA, prepares multi-vector problem data, runs both CPU and GPU multi-RHS TDMA solvers,
 * and compares the results for accuracy.
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @return int Exit code (0 for success)
 */
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Problem size: nx is the number of systems (tridiagonal equations) per process
    const int nx = 8, ny = 40, nz = 160;
    const int N = nx * ny * nz;

    // Tridiagonal coefficients (per system) and right-hand sides
    std::vector<double> a_h(nx, -1.0);
    std::vector<double> b_h(nx,  4.0);
    std::vector<double> c_h(nx, -1.0);
    std::vector<double> d_h(N);

    // Fill the right-hand sides with a known function (sine)
    for (int i = 0; i < N; i++) {
        d_h[i] = std::sin(i);
    }

    // Initialize CUDA environment and report CUDA-aware MPI support
    cudaEnv::initialize();

    if (cudaEnv::isCudaAwareMPI()) {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is available." << std::endl;
    } else {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is NOT available." << std::endl;
    }

    // =====[ Allocate device (GPU) memory ]=====
    double *a_d, *b_d, *c_d, *d_d;
    cudaMalloc(&a_d, nx * sizeof(double));
    cudaMalloc(&b_d, nx * sizeof(double));
    cudaMalloc(&c_d, nx * sizeof(double));
    cudaMalloc(&d_d, N * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(a_d, a_h.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // =====[ CPU reference solve: many RHS TDMA, cyclic boundary ]=====
    PaScaL_TDMA::PTDMAPlanManyRHS px_many;
    px_many.create(nx, ny * nz, MPI_COMM_WORLD, PaScaL_TDMA::TDMAType::Cyclic);
    PaScaL_TDMA::PTDMASolverManyRHS::solve(px_many, a_h, b_h, c_h, d_h);
    px_many.destroy();

    // =====[ GPU solve: many RHS TDMA, cyclic boundary ]=====
    cuPaScaL_TDMA::cuPTDMAPlanManyRHS px_cuMany;
    px_cuMany.create(nx, ny, nz, MPI_COMM_WORLD, cuPaScaL_TDMA::TDMAType::Cyclic);
    cuPaScaL_TDMA::cuPTDMASolverManyRHS::cuSolve(px_cuMany, a_d, b_d, c_d, d_d);
    px_cuMany.destroy();

    // Copy the computed solution from device back to host
    std::vector<double> d_h_out(N);
    cudaMemcpy(d_h_out.data(), d_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    // =====[ Compute and print total error ]=====
    double error = 0.0;
    for (int i = 0; i < N; i++) {
        error += std::abs(d_h[i] - d_h_out[i]);
    }
    if (rank == 0)
        std::cout << "Total error: " << error << std::endl;

    // Free device memory and finalize MPI
    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d); cudaFree(d_d);
    MPI_Finalize();

    return 0;
}