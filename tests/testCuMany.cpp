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

TEST(cuPaScaL_TDMA_many, Solve) {

    // Read from global argc/argv (GoogleTest doesn't pass arguments to TEST directly)
    extern int g_argc;
    extern char** g_argv;

    if (g_argc != 4)
        throw std::runtime_error("Usage: testMany <nx_sub> <ny> <nz>");

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

    const int N = nx_sub * ny * nz;

    std::vector<double> h_a(N, a_lower);
    std::vector<double> h_b(N,  a_diag);
    std::vector<double> h_c(N, a_upper);
    std::vector<double> h_d(N);

    cudaEnv::initialize();

    if (cudaEnv::isCudaAwareMPI()) {
        if (is_root) std::cout << "[INFO] CUDA-Aware MPI is available." << std::endl;
    } else {
        if (is_root) std::cout << "[INFO] CUDA-Aware MPI is NOT available." << std::endl;
    }

    for (int i = 0; i < N; i++) {
        h_d[i] = std::sin(i);
    }

    // GPU memory allocation
    double *d_a, *d_b, *d_c, *d_d;
    cudaMalloc((void**)&d_a, N * sizeof(double));
    cudaMalloc((void**)&d_b, N * sizeof(double));
    cudaMalloc((void**)&d_c, N * sizeof(double));
    cudaMalloc((void**)&d_d, N * sizeof(double));

    cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // CPU reference
    PaScaL_TDMA::PTDMAPlanMany px_many;
    px_many.create(nx_sub, ny*nz, MPI_COMM_WORLD, PaScaL_TDMA::TDMAType::Standard);
    PaScaL_TDMA::PTDMASolverMany::solve(px_many, h_a, h_b, h_c, h_d);
    px_many.destroy();

    // GPU value
    cuPaScaL_TDMA::cuPTDMAPlanMany px_cuMany;
    px_cuMany.create(nx_sub, ny, nz, MPI_COMM_WORLD, cuPaScaL_TDMA::TDMAType::Standard);
    cuPaScaL_TDMA::cuPTDMASolverMany::cuSolve(px_cuMany, d_a, d_b, d_c, d_d);
    px_cuMany.destroy();

    std::vector<double> h_d_out(N);
    cudaMemcpy(h_d_out.data(), d_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    double error = 0.0;
    for (int i = 0; i < N; i++) {
        error += std::abs(h_d[i] - h_d_out[i]);
    }

    if(is_root) std::cout << "Total error: " << error << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_d[i], h_d_out[i], tolerance) << 
            "Mismatch at (i, j, k) = ( " << (int)(i/(ny*nz)) << ", " << (int)(i/nz)%ny << ", " << i%nz <<" )" << std::endl;
    }
}

int g_argc;
char** g_argv;

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    g_argc = argc;
    g_argv = argv;
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}