/**
 * @file testCuManyRHS.cpp
 * @brief Unit test for GPU-accelerated distributed TDMA (many right-hand sides, multi-vector) using cuPaScaL_TDMA, MPI, and GoogleTest.
 *
 * This test compares the results of the CPU (PaScaL_TDMA) and GPU (cuPaScaL_TDMA) multi-RHS solvers for consistency.
 * It checks if the GPU-accelerated multi-RHS TDMA solver produces numerically equivalent results to the CPU version.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "PaScaL_TDMA.cuh"
#include "PaScaL_TDMA.hpp"
#include "cudaEnv.hpp"

constexpr double tolerance = 1e-12;
constexpr double a_diag = 10.0;
constexpr double a_upper = -1.0;
constexpr double a_lower = -1.0;

/**
 * @test
 * @brief GPU-accelerated distributed TDMA (many RHS): compare CPU and GPU solver results.
 *
 * Reads problem size from command line, allocates/initializes data, solves with both CPU and GPU,
 * and checks the results for numerical agreement.
 *
 * Command-line arguments:
 * - nx_sub: Number of grid points in x direction (per process)
 * - ny:     Number of grid points in y direction
 * - nz:     Number of grid points in z direction
 */
TEST(cuPaScaL_TDMA_manyRHS, Solve) {

    // Use global argc/argv as GoogleTest does not pass them directly
    extern int g_argc;
    extern char** g_argv;

    if (g_argc != 4)
        throw std::runtime_error("Usage: testManyRHS <nx_sub> <ny> <nz>");

    const int nx_sub = std::stoi(g_argv[1]);
    const int ny = std::stoi(g_argv[2]);
    const int nz = std::stoi(g_argv[3]);
    ::testing::Test::RecordProperty("nx_sub", nx_sub);
    ::testing::Test::RecordProperty("ny", ny);
    ::testing::Test::RecordProperty("nz", nz);

    if (nx_sub < 10 || nx_sub > 10)
        throw std::runtime_error("Recommendation of 10 <= nx_sub <= 100");

    if (ny < 10 || ny > 100)
        throw std::runtime_error("Recommendation of 10 <= ny <= 100");

    if (nz < 10 || nz > 100)
        throw std::runtime_error("Recommendation of 10 <= nz <= 100");

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    bool is_root = (rank == 0);

    // Host coefficient arrays: note only size nx_sub for a/b/c, but size N for d
    const int N = nx_sub * ny * nz;
    std::vector<double> a_h(nx_sub, a_lower);
    std::vector<double> b_h(nx_sub,  a_diag);
    std::vector<double> c_h(nx_sub, a_upper);
    std::vector<double> d_h(N);

    // Initialize CUDA environment and print CUDA-aware MPI availability
    cudaEnv::initialize();

    if (cudaEnv::isCudaAwareMPI()) {
        if (is_root) std::cout << "[INFO] CUDA-Aware MPI is available." << std::endl;
    } else {
        if (is_root) std::cout << "[INFO] CUDA-Aware MPI is NOT available." << std::endl;
    }

    // Fill the right-hand side with a known function (e.g., sine)
    for (int i = 0; i < N; i++) {
        d_h[i] = std::sin(i);
    }

    // Allocate device (GPU) memory and copy values in host memory
    double *a_d, *b_d, *c_d, *d_d;
    cudaMalloc(&a_d, nx_sub * sizeof(double));
    cudaMalloc(&b_d, nx_sub * sizeof(double));
    cudaMalloc(&c_d, nx_sub * sizeof(double));
    cudaMalloc(&d_d, N * sizeof(double));

    cudaMemcpy(a_d, a_h.data(), nx_sub * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), nx_sub * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h.data(), nx_sub * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // =====[ CPU reference solve ]=====
    PaScaL_TDMA::PTDMAPlanManyRHS px_many;
    px_many.create(nx_sub, ny*nz, MPI_COMM_WORLD, PaScaL_TDMA::TDMAType::Standard);
    PaScaL_TDMA::PTDMASolverManyRHS::solve(px_many, a_h, b_h, c_h, d_h);
    px_many.destroy();

    // =====[ GPU solve ]=====
    cuPaScaL_TDMA::cuPTDMAPlanManyRHS px_cuMany;
    px_cuMany.create(nx_sub, ny, nz, MPI_COMM_WORLD, cuPaScaL_TDMA::TDMAType::Standard);
    cuPaScaL_TDMA::cuPTDMASolverManyRHS::cuSolve(px_cuMany, a_d, b_d, c_d, d_d);
    px_cuMany.destroy();

    std::vector<double> d_h_out(N);
    cudaMemcpy(d_h_out.data(), d_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Compare results: accumulate total error
    double error = 0.0;
    for (int i = 0; i < N; i++) {
        error += std::abs(d_h[i] - d_h_out[i]);
    }

    if(is_root) std::cout << "Total error: " << error << std::endl;

    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d); cudaFree(d_d);

    // Assert: Each value within tolerance
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(d_h[i], d_h_out[i], tolerance) << 
            "Mismatch at (i, j, k) = ( " << (int)(i/(ny*nz)) << ", " << (int)(i/nz)%ny << ", " << i%nz <<" )" << std::endl;
    }
}

// Global variables for GoogleTest argument passing
int g_argc;
char** g_argv;

/**
 * @brief Main entry point for MPI+GoogleTest runs.
 *
 * Initializes MPI, passes arguments to tests, runs all tests, and finalizes MPI.
 */
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    g_argc = argc;
    g_argv = argv;
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}