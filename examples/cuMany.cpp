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
    
    const int nx = 4, ny = 40, nz = 160;
    const int N = nx * ny * nz;

    std::vector<double> a_h(N, -1.0);
    std::vector<double> b_h(N,  4.0);
    std::vector<double> c_h(N, -1.0);
    std::vector<double> d_h(N);

    cudaEnv::initialize();

    if (cudaEnv::isCudaAwareMPI()) {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is available." << std::endl;
    } else {
        if (!rank) std::cout << "[INFO] CUDA-Aware MPI is NOT available." << std::endl;
    }

    for (int i = 0; i < N; i++) {
        d_h[i] = std::sin(i);
    }

    // GPU 메모리 할당
    double *a_d, *b_d, *c_d, *d_d;
    cudaMalloc((void**)&a_d, N * sizeof(double));
    cudaMalloc((void**)&b_d, N * sizeof(double));
    cudaMalloc((void**)&c_d, N * sizeof(double));
    cudaMalloc((void**)&d_d, N * sizeof(double));

    cudaMemcpy(a_d, a_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // CPU 참조 해 계산

    PaScaL_TDMA::PTDMAPlanMany px_many;
    px_many.create(nx, ny*nz, MPI_COMM_WORLD, PaScaL_TDMA::TDMAType::Standard);
    PaScaL_TDMA::PTDMASolverMany::solve(px_many, a_h, b_h, c_h, d_h);
    px_many.destroy();

    cuPaScaL_TDMA::cuPTDMAPlanMany px_cuMany;
    px_cuMany.create(nx, ny, nz, MPI_COMM_WORLD, cuPaScaL_TDMA::TDMAType::Standard);
    cuPaScaL_TDMA::cuPTDMASolverMany::cuSolve(px_cuMany, a_d, b_d, c_d, d_d);
    px_cuMany.destroy();

    std::vector<double> d_h_out(N);
    cudaMemcpy(d_h_out.data(), d_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    double error = 0.0;
    for (int i = 0; i < N; i++) {
        error += std::abs(d_h[i] - d_h_out[i]);
    }

    if(!rank) std::cout << "Total error: " << error << std::endl;

    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d); cudaFree(d_d);
    MPI_Finalize();

    return 0;
}