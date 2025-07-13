#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "PaScaL_TDMA.cuh"
#include "PaScaL_TDMA.hpp"
#include "cudaEnv.hpp"

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int nx = 8, ny = 40, nz = 160;
    const int N = nx * ny * nz;

    std::vector<double> a_h(nx, -1.0);
    std::vector<double> b_h(nx,  4.0);
    std::vector<double> c_h(nx, -1.0);
    std::vector<double> d_h(N);

    for (int i = 0; i < N; i++) {
        d_h[i] = std::sin(i);
    }

    cudaEnv::initialize();

    if (cudaEnv::isCudaAwareMPI()) {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is available." << std::endl;
    } else {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is NOT available." << std::endl;
    }

    // GPU 메모리 할당
    double *a_d, *b_d, *c_d, *d_d;
    cudaMalloc(&a_d, nx * sizeof(double));
    cudaMalloc(&b_d, nx * sizeof(double));
    cudaMalloc(&c_d, nx * sizeof(double));
    cudaMalloc(&d_d, N * sizeof(double));

    cudaMemcpy(a_d, a_h.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // CPU 참조 해 계산
    PaScaL_TDMA::PTDMAPlanManyRHS px_many;
    px_many.create(nx, ny * nz, MPI_COMM_WORLD, PaScaL_TDMA::TDMAType::Cyclic);
    PaScaL_TDMA::PTDMASolverManyRHS::solve(px_many, a_h, b_h, c_h, d_h);
    px_many.destroy();

    // 커널 호출
    cuPaScaL_TDMA::cuPTDMAPlanManyRHS px_cuMany;
    px_cuMany.create(nx, ny, nz, MPI_COMM_WORLD, cuPaScaL_TDMA::TDMAType::Cyclic);
    cuPaScaL_TDMA::cuPTDMASolverManyRHS::cuSolve(px_cuMany, a_d, b_d, c_d, d_d);
    px_cuMany.destroy();

    // 결과 복사 및 비교
    std::vector<double> d_h_out(N);
    cudaMemcpy(d_h_out.data(), d_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    double error = 0.0;
    for (int i = 0; i < N; i++) {
        error += std::abs(d_h[i] - d_h_out[i]);
    }
    if (rank == 0)
        std::cout << "Total error: " << error << std::endl;

    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d); cudaFree(d_d);
    MPI_Finalize();

    return 0;
}