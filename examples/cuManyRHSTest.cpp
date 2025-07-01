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

    std::vector<double> h_a(nx, -1.0);
    std::vector<double> h_b(nx,  4.0);
    std::vector<double> h_c(nx, -1.0);
    std::vector<double> h_d(N);

    for (int i = 0; i < N; i++) {
        h_d[i] = std::sin(i);
    }

    // for (int j = 0; j < ny * nz; j++)
    //         h_d[j] = 2.0;
    // for (int i = ny * nz; i < (nx - 1) * ny * nz; i += ny * nz) {
    //     for (int j = 0; j < ny * nz; j++)
    //         h_d[i + j] = 2.0;
    // }
    // int i = (nx - 1) * ny * nz;
    // for (int j = 0; j < ny * nz; j++)
    //     h_d[i + j] = 2.0;

    std::vector<double> h_d_ref = h_d;

    // for (int i = 0; i < nx * ny * nz; i+=ny*nz) {
    //     for (int j = 0; j < ny*nz; j++) std::cout << h_d_ref[i+j] << ' ';
    //     std::cout<<std::endl;
    // }

    cudaEnv::initialize();

    if (cudaEnv::isCudaAwareMPI()) {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is available." << std::endl;
    } else {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is NOT available." << std::endl;
    }

    // GPU 메모리 할당
    double *d_a, *d_b, *d_c, *d_d;
    cudaMalloc(&d_a, nx * sizeof(double));
    cudaMalloc(&d_b, nx * sizeof(double));
    cudaMalloc(&d_c, nx * sizeof(double));
    cudaMalloc(&d_d, N * sizeof(double));

    cudaMemcpy(d_a, h_a.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // CPU 참조 해 계산
    PaScaL_TDMA::PTDMAPlanManyRHS px_many;
    px_many.create(nx, ny * nz, MPI_COMM_WORLD, PaScaL_TDMA::TDMAType::Cyclic);
    PaScaL_TDMA::PTDMASolverManyRHS::solve(px_many, h_a, h_b, h_c, h_d_ref);
    px_many.destroy();

    // 커널 호출
    cuPaScaL_TDMA::cuPTDMAPlanManyRHS px_cuMany;
    px_cuMany.create(nx, ny, nz, MPI_COMM_WORLD, cuPaScaL_TDMA::TDMAType::Cyclic);
    cuPaScaL_TDMA::cuPTDMASolverManyRHS::cuSolve(px_cuMany, d_a, d_b, d_c, d_d);
    px_cuMany.destroy();

    // 결과 복사 및 비교
    std::vector<double> h_d_out(N);
    cudaMemcpy(h_d_out.data(), d_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < nx * ny * nz; i+=ny*nz) {
    //     for (int j = 0; j < ny*nz; j++) std::cout << h_d_out[i+j] << ' ';
    //     std::cout<<std::endl;
    // }

    double error = 0.0;
    for (int i = 0; i < N; i++) {
        error += std::abs(h_d_ref[i] - h_d_out[i]);
    }
    if (rank == 0)
        std::cout << "Total error: " << error << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    MPI_Finalize();

    return 0;
}