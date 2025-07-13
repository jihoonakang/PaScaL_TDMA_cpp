/**
 * @file cuMany.cpp
 * @brief Example for GPU-accelerated distributed TDMA (many right-hand sides) using cuPaScaL_TDMA and MPI.
 *
 * This example demonstrates how to set up, solve, and compare the results of CPU and GPU solvers for a large TDMA system in parallel, using CUDA-aware MPI.
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
 * Initializes MPI and CUDA, prepares problem data, runs both CPU and GPU TDMA solvers, 
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
    
    // Problem size settings
    const int nx = 4, ny = 40, nz = 160;
    const int N = nx * ny * nz;

    // Host-side tridiagonal system coefficients and right-hand side
    std::vector<double> a_h(N, -1.0);
    std::vector<double> b_h(N,  4.0);
    std::vector<double> c_h(N, -1.0);
    std::vector<double> d_h(N);

    // Initialize CUDA environment (check CUDA-aware MPI)
    cudaEnv::initialize();

    if (cudaEnv::isCudaAwareMPI()) {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is available." << std::endl;
    } else {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is NOT available." << std::endl;
    }

    // Fill the right-hand side with a known function (sine)
    for (int i = 0; i < N; i++) {
        d_h[i] = std::sin(i);
    }

    // =====[ Allocate device (GPU) memory ]=====
    double *a_d, *b_d, *c_d, *d_d;
    cudaMalloc((void**)&a_d, N * sizeof(double));
    cudaMalloc((void**)&b_d, N * sizeof(double));
    cudaMalloc((void**)&c_d, N * sizeof(double));
    cudaMalloc((void**)&d_d, N * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(a_d, a_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // =====[ CPU reference solve ]=====
    PaScaL_TDMA::PTDMAPlanMany px_many;
    px_many.create(nx, ny*nz, MPI_COMM_WORLD, PaScaL_TDMA::TDMAType::Standard);
    PaScaL_TDMA::PTDMASolverMany::solve(px_many, a_h, b_h, c_h, d_h);
    px_many.destroy();

    // =====[ GPU solve ]=====
    cuPaScaL_TDMA::cuPTDMAPlanMany px_cuMany;
    px_cuMany.create(nx, ny, nz, MPI_COMM_WORLD, cuPaScaL_TDMA::TDMAType::Standard);
    cuPaScaL_TDMA::cuPTDMASolverMany::cuSolve(px_cuMany, a_d, b_d, c_d, d_d);
    px_cuMany.destroy();

    // Copy the computed solution from device back to host
    std::vector<double> d_h_out(N);
    cudaMemcpy(d_h_out.data(), d_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    // =====[ Compute and print total error ]=====
    double error = 0.0;
    for (int i = 0; i < N; i++) {
        error += std::abs(d_h[i] - d_h_out[i]);
    }

    if(!rank) std::cout << "Total error: " << error << std::endl;

    // Free device memory and finalize MPI
    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d); cudaFree(d_d);
    MPI_Finalize();

    return 0;
}