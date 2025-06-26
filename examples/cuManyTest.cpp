#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "TDMASolver.hpp"
#include "TDMASolver.cuh"
#include "PaScaL_TDMA.cuh"
#include "PaScaL_TDMA.hpp"

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    const int nx = 4, ny = 40, nz = 160;
    const int N = nx * ny * nz;

    std::vector<double> h_a(N, -1.0);
    std::vector<double> h_b(N,  4.0);
    std::vector<double> h_c(N, -1.0);
    std::vector<double> h_d(N);

    for (int i = 0; i < N; i++) {
        h_d[i] = std::sin(i);
    }

    // for (int j = 0; j < ny * nz; j++)
    //         h_d[j] = 8.0;
    // for (int i = ny * nz; i < (nx - 1) * ny * nz; i += ny * nz) {
    //     for (int j = 0; j < ny * nz; j++)
    //         h_d[i + j] = 8.0;
    // }
    // int i = (nx - 1) * ny * nz;
    // for (int j = 0; j < ny * nz; j++)
    //     h_d[i + j] = 8.0;

    // GPU 메모리 할당
    double *d_a, *d_b, *d_c, *d_d;
    cudaMalloc((void**)&d_a, N * sizeof(double));
    cudaMalloc((void**)&d_b, N * sizeof(double));
    cudaMalloc((void**)&d_c, N * sizeof(double));
    cudaMalloc((void**)&d_d, N * sizeof(double));

    cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // CPU 참조 해 계산

    PaScaL_TDMA::PTDMAPlanMany px_many;
    px_many.create(nx, ny*nz, MPI_COMM_WORLD, PaScaL_TDMA::TDMAType::Standard);
    PaScaL_TDMA::PTDMASolverMany::solve(px_many, h_a, h_b, h_c, h_d);
    px_many.destroy();

    cuPaScaL_TDMA::cuPTDMAPlanMany px_cuMany;
    px_cuMany.create(nx, ny, nz, MPI_COMM_WORLD, cuPaScaL_TDMA::TDMAType::Standard);
    cuPaScaL_TDMA::cuPTDMASolverMany::cuSolve(px_cuMany, d_a, d_b, d_c, d_d);
    px_cuMany.destroy();

    std::vector<double> h_d_out(N);
    cudaMemcpy(h_d_out.data(), d_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    double error = 0.0;
    for (int i = 0; i < N; i++) {
        error += std::abs(h_d[i] - h_d_out[i]);
    }

    if(!rank) std::cout << "Total error: " << error << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    MPI_Finalize();

    return 0;
}